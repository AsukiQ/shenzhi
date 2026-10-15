"""Resumable, online canonical-paper import. Preview by default; --apply writes."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sqlite3
import sys
from urllib import request
import zipfile

from funding_cleanup import clean_funding_name
from paper_search import PaperSearchIndex, _fts_text

ROOT = Path(__file__).resolve().parent
VERSION = 'incremental-v1'
KEYS = {'Paper': 'paper_id', 'Author': 'author_id', 'Topic': 'topic_id',
        'Method': 'method_id', 'Institution': 'institution_id', 'Venue': 'venue_id',
        'Conference': 'conference_id', 'Funding': 'funding_id'}
ENDPOINTS = {
    'AUTHORED_BY': ('Paper', 'Author'), 'HAS_TOPIC': ('Paper', 'Topic'),
    'PUBLISHED_IN': ('Paper', 'Conference'), 'CITES': ('Paper', 'Paper'),
    'USES_AS_BASELINE': ('Paper', 'Method'), 'PROPOSES': ('Paper', 'Method'),
    'FUNDED_BY': ('Paper', 'Funding'), 'HAS_AFFILIATION': ('Paper', 'Institution'),
    'AFFILIATED_WITH': ('Author', 'Institution'), 'PART_OF': ('Conference', 'Venue'),
}
MANAGED = [t for t, (source, _) in ENDPOINTS.items() if source == 'Paper']
ARRAYS = {'authors': 'paper_authors', 'keywords': 'paper_keywords',
          'subjects': 'paper_subjects', 'funding': 'paper_funding'}
SECTION = re.compile(r'^(\d+(\.\d+)*\.?\s+)?(introduction|abstract|related work|background|conclusions?|references|preliminaries)$', re.I)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def image_key(member):
    path = PurePosixPath(member)
    stem = re.sub(r'(_\d{8}_\d{6})?_canonical$', '', path.stem)
    stem = re.sub(r'_minimal$', '', stem)
    return path.parent.name.removeprefix('canonical_'), stem


def stage_images(folder, archives):
    """Save matched image bytes in the batch so recovery needs no source ZIPs."""
    conn = sqlite3.connect(folder / 'journal.sqlite3')
    keys = {}
    for seq, record in conn.execute('SELECT seq,record FROM jobs'):
        keys[image_key(json.loads(record)['source']['member'])] = seq
    matched = set()
    try:
        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    path = PurePosixPath(info.filename)
                    if info.is_dir() or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.svg'}:
                        continue
                    if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                        raise ValueError(f'Unsafe image path: {info.filename}')
                    seq = keys.get(image_key(info.filename))
                    if seq is None:
                        continue
                    if seq in matched:
                        raise ValueError('More than one image matched a paper; provide only its primary image')
                    if info.file_size > 20 * 1024 * 1024:
                        raise ValueError('Primary image exceeds 20 MiB')
                    matched.add(seq)
                    payload = archive.read(info)
                    sha = hashlib.sha256(payload).hexdigest()
                    asset = folder / 'assets' / (sha + path.suffix.lower())
                    asset.parent.mkdir(exist_ok=True)
                    with asset.open('wb') as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    plan = json.loads(conn.execute('SELECT plan FROM jobs WHERE seq=?', (seq,)).fetchone()[0])
                    plan['image'] = dict(asset=str(asset.relative_to(folder)), sha256=sha,
                                         image_path='images/incremental/' + asset.name, caption=None)
                    plan['doc']['image_id'] = '/' + plan['image']['image_path']
                    plan['revision'] = digest({k: v for k, v in plan.items() if k != 'revision'})
                    conn.execute('UPDATE jobs SET plan=? WHERE seq=?', (encoded(plan), seq))
        conn.commit()
    finally:
        conn.close()


def publish_image(plan, folder, db):
    if 'image' not in plan:
        return
    image = plan['image']
    source = folder / image['asset']
    if file_hash(source) != image['sha256']:
        raise RuntimeError('Staged image checksum mismatch')
    destination = Path(db).parent.parent / image['image_path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + '.tmp')
    with temp.open('wb') as handle:
        handle.write(source.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(destination)
    manifest_path = Path(db).with_name('paper_images.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest[plan['pid']] = [dict(image_id=image['sha256'], image_path=image['image_path'], caption=image['caption'])]
    atomic_json(manifest_path, manifest)


def strings(value):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError('Expected an array of strings')
    return list(dict.fromkeys(v.strip() for v in value if v.strip()))


def norm(value):
    # Keep non-Latin names; ASCII-only normalization collapses Chinese names.
    return ''.join(c for c in str(value).casefold() if c.isalnum())


def primitives(props):
    # IDs always come from the graph envelope, never an input property override.
    return {k: v for k, v in props.items()
            if k not in {*KEYS.values(), 'id', 'incremental_revision', 'incremental_batch'}
            and isinstance(v, (str, int, float, bool)) and v is not None}


def iter_graphs(paths):
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.namelist()):
                    if member.endswith('.json'):
                        yield str(path), member, json.loads(archive.read(member))
        elif path.suffix.lower() == '.rar':
            try:
                from unrar.cffi import rarfile
            except ImportError as exc:
                raise ValueError('RAR requires unrar.cffi; alternatively provide ZIP/JSON') from exc
            archive = rarfile.RarFile(str(path))
            for member in sorted(archive.namelist()):
                if member.endswith('.json'):
                    yield str(path), member, json.loads(archive.read(member))
        elif path.suffix.lower() == '.jsonl':
            with path.open(encoding='utf-8') as handle:
                for number, line in enumerate(handle, 1):
                    if line.strip():
                        yield str(path), str(number), json.loads(line)
        elif path.suffix.lower() == '.json':
            yield str(path), path.name, json.loads(path.read_text(encoding='utf-8'))
        else:
            raise ValueError(f'Unsupported input format: {path}')


def normalize(data, archive, member):
    graph = data.get('graph', data)
    if not isinstance(graph, dict) or 'nodes' not in graph:
        raise ValueError(f'Not a canonical graph: {archive}:{member}')
    if 'edges' not in graph and 'relationships' not in graph:
        raise ValueError('A graph must explicitly contain edges/relationships (an empty list is allowed)')
    nodes = {}
    for node in graph['nodes']:
        nid, label = node.get('id'), node.get('label')
        if not isinstance(nid, str) or not nid.strip() or label not in KEYS:
            raise ValueError(f'Unsupported label or empty ID: {label!r} / {nid!r}')
        if nid in nodes:
            raise ValueError(f'Duplicate node ID: {nid}')
        props = node.get('properties') or {}
        if not isinstance(props, dict):
            raise ValueError(f'Invalid properties: {nid}')
        nodes[nid] = dict(id=nid, label=label, props=props)
    edges = []
    for edge in graph.get('edges', graph.get('relationships', [])):
        s, t, typ = edge.get('source'), edge.get('target'), edge.get('type')
        if s not in nodes or t not in nodes:
            raise ValueError(f'Missing relationship endpoint: {s} -> {t}')
        if ENDPOINTS.get(typ) != (nodes[s]['label'], nodes[t]['label']):
            raise ValueError(f'Unsupported relationship: {typ}: {nodes[s]["label"]} -> {nodes[t]["label"]}')
        edges.append(dict(s=s, t=t, type=typ, props=primitives(edge.get('properties') or {})))
    papers = [n for n in nodes.values() if n['label'] == 'Paper']
    main_id = data.get('main_paper_id') or graph.get('main_paper_id')
    if main_id:
        candidates = [n for n in papers if n['id'] == main_id]
    else:
        candidates = [n for n in papers if n['props'].get('external') is False]
        if not candidates:
            owner_ids = {e['s'] for e in edges if e['type'] in ('PUBLISHED_IN', 'AUTHORED_BY', 'HAS_TOPIC', 'PROPOSES')}
            candidates = [n for n in papers if n['id'] in owner_ids and n['props'].get('external') is not True]
        if not candidates and len(papers) == 1 and papers[0]['props'].get('external') is not True:
            candidates = papers
    if len(candidates) != 1:
        raise ValueError('Cannot identify one main paper; supply main_paper_id explicitly')
    main = candidates[0]
    title = re.sub(r'\s+', ' ', str(main['props'].get('title') or '')).strip()
    if len(title) < 12 or SECTION.fullmatch(title):
        raise ValueError(f'Invalid main-paper title: {title!r}')
    main['props']['title'] = title
    for field in ('authors', 'keywords', 'subjects', 'funding'):
        if field in main['props']:
            main['props'][field] = strings(main['props'][field])
    # Relationships owned by other papers would make a per-paper snapshot unsafe.
    if any(nodes[e['s']]['label'] == 'Paper' and e['s'] != main['id'] for e in edges):
        raise ValueError('A per-paper graph may only contain outgoing relationships of its main paper')
    return dict(pid=main['id'], nodes=list(nodes.values()), edges=edges,
                source={'archive': Path(archive).name, 'member': member})


class GraphStore:
    def __init__(self, uri, user, password, database='neo4j', timeout=60):
        self.identity = uri.rstrip('/') + '/db/' + database
        self.endpoint = self.identity + '/tx/commit'
        self.authorization = 'Basic ' + base64.b64encode(f'{user}:{password}'.encode()).decode()
        self.timeout = timeout

    def transaction(self, statements):
        payload = {'statements': [dict(statement=s, parameters=p) for s, p in statements]}
        req = request.Request(self.endpoint, encoded(payload).encode(),
                              {'Content-Type': 'application/json', 'Authorization': self.authorization})
        with request.urlopen(req, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get('errors'):
            raise RuntimeError('Neo4j transaction failed: ' + encoded(result['errors']))
        return [[dict(zip(part['columns'], row['row'])) for row in part['data']]
                for part in result['results']]

    def query(self, statement, parameters=None):
        return self.transaction([(statement, parameters or {})])[0]

    def preflight(self):
        constraints = self.query('SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties RETURN type, labelsOrTypes, properties')
        present = {(r['labelsOrTypes'][0], r['properties'][0]) for r in constraints
                   if r['type'] == 'UNIQUENESS' and len(r['labelsOrTypes']) == len(r['properties']) == 1}
        missing = [(label, key) for label, key in KEYS.items() if (label, key) not in present]
        if missing:
            raise RuntimeError(f'Node ID uniqueness constraints required before import: {missing}')

    def authors(self, names):
        return self.query('''MATCH (a:Author) WHERE a.name IN $names
            OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution)
            RETURN a.name AS name, a.author_id AS id, collect(i.name) AS institutions''', {'names': names}) if names else []

    def snapshot(self, plan):
        groups = defaultdict(list)
        for node in plan['nodes']:
            groups[node['label']].append(node['id'])
        statements = []
        for label, ids in sorted(groups.items()):
            statements.append((f'''MATCH (n:{label}) WHERE n.{KEYS[label]} IN $ids
                OPTIONAL MATCH (n)-[r]->(t)
                RETURN labels(n) AS labels, properties(n) AS props,
                collect(CASE WHEN r IS NULL THEN null ELSE {{type:type(r), props:properties(r),
                    target_labels:labels(t), target_props:properties(t)}} END) AS outgoing''', {'ids': ids}))
        return self.transaction(statements)

    def write(self, plan, batch_id):
        groups = defaultdict(list)
        for node in plan['nodes']:
            groups[(node['label'], node['id'] == plan['pid'])].append(node)
        statements = []
        for (label, is_main), rows in sorted(groups.items()):
            if is_main:
                update = 'SET n += row.props, n.external=false, n.incremental_revision=$revision, n.incremental_batch=$batch'
            elif label == 'Paper':
                # Reference stubs must never downgrade or overwrite a complete paper.
                update = 'ON CREATE SET n += row.props, n.external=true'
            else:
                update = 'SET n += row.props'
            statements.append((f'UNWIND $rows AS row MERGE (n:{label} {{{KEYS[label]}:row.id}}) {update}',
                               dict(rows=rows, revision=plan['revision'], batch=batch_id)))
        if plan['relation_mode'] == 'replace':
            expected = [dict(type=e['type'], id=e['t'], key=KEYS[e['tl']])
                        for e in plan['edges'] if e['s'] == plan['pid']]
            statements.append(('''MATCH (p:Paper {paper_id:$pid})-[r]->(t)
                WHERE type(r) IN $types AND NOT any(e IN $expected
                    WHERE e.type=type(r) AND t[e.key]=e.id)
                DELETE r''', dict(pid=plan['pid'], types=MANAGED, expected=expected)))
        relationships = defaultdict(list)
        for edge in plan['edges']:
            relationships[(edge['type'], edge['sl'], edge['tl'])].append(edge)
        for (typ, sl, tl), rows in sorted(relationships.items()):
            statements.append((f'''UNWIND $rows AS row
                MATCH (s:{sl} {{{KEYS[sl]}:row.s}}), (t:{tl} {{{KEYS[tl]}:row.t}})
                MERGE (s)-[r:{typ}]->(t) SET r += row.props''', {'rows': rows}))
        statements.append(('''MATCH (p:Paper {paper_id:$pid}) RETURN p.external AS external,
            p.incremental_revision AS revision''', {'pid': plan['pid']}))
        result = self.transaction(statements)[-1]
        if result != [{'external': False, 'revision': plan['revision']}]:
            raise RuntimeError('Graph verification failed')


def make_plan(record, previous, graph, mode):
    nodes = {n['id']: dict(n, props=dict(n['props'])) for n in record['nodes']}
    edges = [dict(e, props=dict(e['props'])) for e in record['edges']]
    pid = record['pid']
    props = nodes[pid]['props']
    doc = dict(previous or {})
    for key in ('source_id', 'title', 'abstract', 'conference', 'year', 'image_id'):
        doc.setdefault(key, None if key not in ('title', 'abstract') else '')
    for field in ARRAYS:
        doc[field] = json.loads(doc.pop(field + '_json', '[]')) if field + '_json' in doc else doc.get(field, [])
    doc['title'] = props['title']
    for field in ('abstract', 'year', 'image_id'):
        if field in props:
            doc[field] = props[field]
    if doc['year'] not in (None, ''):
        doc['year'] = int(doc['year'])
    else:
        doc['year'] = None
    for field in ('arxiv_id', 'doi'):
        if props.get(field):
            doc['source_id'] = props[field]
            break
    conferences = [nodes[e['t']]['props'] for e in edges if e['s'] == pid and e['type'] == 'PUBLISHED_IN']
    if len(conferences) > 1:
        raise ValueError('Multiple PUBLISHED_IN targets for one main paper')
    if conferences:
        conf = conferences[0]
        doc['conference'] = props.get('venue') or props.get('conference') or conf.get('venue') or conf.get('name')
        if 'year' not in props and conf.get('year') is not None:
            doc['year'] = int(conf['year'])
    elif 'venue' in props or 'conference' in props:
        doc['conference'] = props.get('venue', props.get('conference'))
    elif mode == 'replace':
        doc['conference'] = None
    targets = lambda typ: [nodes[e['t']]['props'].get('name') for e in edges if e['s'] == pid and e['type'] == typ]
    for field, typ in [('authors', 'AUTHORED_BY'), ('subjects', 'HAS_TOPIC'), ('funding', 'FUNDED_BY')]:
        incoming = [v for v in targets(typ) if v]
        # Graph edges are the source of truth for fields also used by graph filters.
        doc[field] = list(dict.fromkeys((doc[field] if mode == 'merge' else []) + incoming))
    if 'keywords' in props:
        doc['keywords'] = props['keywords']
    doc['abstract'] = str(doc['abstract'] or '')
    doc['paper_id'] = pid
    if not doc['abstract'] and not doc['authors']:
        raise ValueError('Main paper has neither abstract nor author relationships')

    # Resolve authors conservatively and persist the chosen IDs before any writes.
    authors = [n for n in nodes.values() if n['label'] == 'Author']
    known = graph.authors(sorted({n['props'].get('name', '') for n in authors}))
    remap = {}
    for author in authors:
        name = str(author['props'].get('name') or '').strip()
        if not name:
            raise ValueError('Author node needs a name')
        institutions = {norm(nodes[e['t']]['props'].get('name', '')) for e in edges
                        if e['s'] == author['id'] and e['type'] == 'AFFILIATED_WITH'} - {''}
        matches = {a['id'] for a in known if a['id'] and a['name'] == name
                   and institutions & {norm(i) for i in a['institutions']}}
        direct = next((a['id'] for a in known if a['id'] == author['id']), None)
        if direct:
            aid = direct
        elif len(matches) == 1:
            aid = next(iter(matches))
        else:
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            fingerprint = f"{name}|{'|'.join(sorted(institutions))}|{pid if not institutions else ''}"
            aid = f'author:derived:{slug}:' + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        remap[author['id']] = aid

    # Apply the same conservative funding cleanup used by full index builds.
    funding_remap = {}
    for node in list(nodes.values()):
        if node['label'] != 'Funding':
            continue
        names, _ = clean_funding_name(node['props'].get('name'))
        if names == [node['props'].get('name')]:
            continue
        funding_remap[node['id']] = []
        for name in names:
            fid = 'funding:normalized:' + hashlib.sha256(name.casefold().strip().encode()).hexdigest()[:24]
            nodes[fid] = dict(id=fid, label='Funding', props={'name': name})
            funding_remap[node['id']].append(fid)
        del nodes[node['id']]
    normalized_edges = []
    for edge in edges:
        for tid in funding_remap.get(edge['t'], [edge['t']]):
            normalized_edges.append(dict(edge, s=remap.get(edge['s'], edge['s']), t=remap.get(tid, tid)))
    edges = normalized_edges
    doc['funding'] = list(dict.fromkeys(n for raw in doc['funding'] for n in clean_funding_name(raw)[0]))
    source = encoded({'archives': {record['source']['archive']: 1}, 'sample_members': [record['source']]})
    main_props = primitives(props)
    main_props.update(title=doc['title'], abstract=doc['abstract'], year=doc['year'], venue=doc['conference'],
                      authors_json=encoded(doc['authors']), keywords_json=encoded(doc['keywords']),
                      properties_json=encoded(props), sources_json=source)
    # Explicit null clears optional metadata; missing scalar fields were preserved above.
    outgoing = []
    seen = set()
    for node in nodes.values():
        nid = remap.get(node['id'], node['id'])
        if (node['label'], nid) in seen:
            continue
        seen.add((node['label'], nid))
        pr = main_props if nid == pid else primitives(node['props'])
        if node['label'] == 'Paper' and nid != pid:
            pr = {k: v for k, v in pr.items() if k in ('title', 'year', 'arxiv_id', 'doi', 'url', 'pdf_url')}
        pr = dict(pr, sources_json=source)
        outgoing.append(dict(id=nid, label=node['label'], props=pr))
    labels = {n['id']: n['label'] for n in outgoing}
    unique_edges = {}
    for edge in edges:
        edge.update(sl=labels[edge['s']], tl=labels[edge['t']])
        edge['props']['sources_json'] = source
        unique_edges[(edge['s'], edge['type'], edge['t'])] = edge
    doc = {k: doc[k] for k in ('paper_id', 'source_id', 'title', 'abstract', 'conference', 'year', 'image_id', *ARRAYS)}
    plan = dict(pid=pid, nodes=outgoing, edges=list(unique_edges.values()), doc=doc, relation_mode=mode)
    plan['revision'] = digest(plan)
    return plan


class SqlStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.conn = PaperSearchIndex(self.path).connect(writable=True)
        self.conn.execute('PRAGMA busy_timeout=30000')
        self.conn.execute('PRAGMA synchronous=FULL')
        # One scan per invocation. paper_id is UNINDEXED in FTS; scanning it once
        # per updated paper made previous replacement scripts quadratic.
        self.conn.execute('CREATE TEMP TABLE import_fts_rows AS SELECT paper_id,rowid AS fts_rowid FROM papers_fts')
        self.conn.execute('CREATE INDEX temp.import_fts_pid ON import_fts_rows(paper_id)')
        self.conn.commit()

    def previous(self, pid):
        row = self.conn.execute('SELECT * FROM papers WHERE paper_id=?', (pid,)).fetchone()
        return dict(row) if row else None

    def snapshot(self, pid):
        return dict(paper=self.previous(pid),
                    fts=[dict(r) for r in self.conn.execute('''SELECT f.rowid AS fts_rowid,f.* FROM papers_fts f
                        JOIN import_fts_rows m ON m.fts_rowid=f.rowid WHERE m.paper_id=?''', (pid,))],
                    relations={table: [dict(r) for r in self.conn.execute(f'SELECT * FROM {table} WHERE paper_id=?', (pid,))]
                               for table in ARRAYS.values()})

    def write(self, plan):
        doc = plan['doc']
        pid = plan['pid']
        with self.conn:
            fields = ['paper_id', 'source_id', 'title', 'abstract', 'conference', 'year',
                      'authors_json', 'keywords_json', 'subjects_json', 'funding_json', 'image_id']
            values = [doc.get(k) if not k.endswith('_json') else encoded(doc[k[:-5]]) for k in fields]
            self.conn.execute(f'''INSERT INTO papers ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})
                ON CONFLICT(paper_id) DO UPDATE SET {','.join(k+'=excluded.'+k for k in fields if k!='paper_id')}''', values)
            self.conn.execute('DELETE FROM papers_fts WHERE rowid IN (SELECT fts_rowid FROM import_fts_rows WHERE paper_id=?)', (pid,))
            self.conn.execute('DELETE FROM import_fts_rows WHERE paper_id=?', (pid,))
            text = (pid, _fts_text(doc['title']), _fts_text(doc['abstract']),
                    _fts_text(' '.join(doc['keywords'])), _fts_text(' '.join(doc['subjects'] + doc['funding'])))
            cursor = self.conn.execute('INSERT INTO papers_fts(paper_id,title,abstract,keywords,subjects) VALUES (?,?,?,?,?)', text)
            self.conn.execute('INSERT INTO import_fts_rows VALUES (?,?)', (pid, cursor.lastrowid))
            for field, table in ARRAYS.items():
                self.conn.execute(f'DELETE FROM {table} WHERE paper_id=?', (pid,))
                self.conn.executemany(f'INSERT OR IGNORE INTO {table}(paper_id,value,value_key,position) VALUES (?,?,?,?)',
                                     [(pid, v, v.casefold(), i) for i, v in enumerate(doc[field])])
            row = self.conn.execute('SELECT paper_id,title,abstract,keywords,subjects FROM papers_fts WHERE rowid=?', (cursor.lastrowid,)).fetchone()
            if tuple(row) != text:
                raise RuntimeError('FTS verification failed')
            actual = self.previous(pid)
            if any(actual[k] != value for k, value in zip(fields, values)):
                raise RuntimeError('SQLite metadata verification failed')

    def close(self):
        self.conn.close()


@contextmanager
def import_lock(db):
    path = Path(str(Path(db).resolve()) + '.incremental.lock')
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another incremental import is running for this database') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Batch:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.conn = sqlite3.connect(self.folder / 'journal.sqlite3')
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA synchronous=FULL')
        self.conn.executescript('''CREATE TABLE IF NOT EXISTS jobs (
            seq INTEGER PRIMARY KEY, pid TEXT NOT NULL UNIQUE, record TEXT NOT NULL,
            plan TEXT, status TEXT NOT NULL DEFAULT 'pending', error TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY, pid TEXT, stage TEXT NOT NULL, at TEXT NOT NULL, error TEXT);''')

    def event(self, pid, stage, error=None):
        with self.conn:
            self.conn.execute('UPDATE jobs SET status=?,error=?,updated_at=? WHERE pid=?', (stage, error, stamp(), pid))
            self.conn.execute('INSERT INTO events(pid,stage,at,error) VALUES (?,?,?,?)', (pid, stage, stamp(), error))

    def counts(self):
        return dict(self.conn.execute('SELECT status,count(*) FROM jobs GROUP BY status').fetchall())

    def run(self, sql, graph):
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        if manifest['db'] != str(sql.path) or manifest['graph'] != graph.identity:
            raise RuntimeError('Resume target differs from staged database/Neo4j target')
        rows = self.conn.execute("SELECT seq FROM jobs WHERE status!='completed' ORDER BY seq").fetchall()
        for number, item in enumerate(rows, 1):
            row = self.conn.execute('SELECT * FROM jobs WHERE seq=?', (item['seq'],)).fetchone()
            pid = row['pid']
            try:
                if row['plan']:
                    plan = json.loads(row['plan'])
                else:
                    plan = make_plan(json.loads(row['record']), sql.previous(pid), graph, manifest['relation_mode'])
                    with self.conn:
                        self.conn.execute('UPDATE jobs SET plan=? WHERE seq=?', (encoded(plan), row['seq']))
                if plan['revision'] != digest({k: v for k, v in plan.items() if k != 'revision'}):
                    raise RuntimeError('Staged plan checksum mismatch')
                backup_path = self.folder / f'before_{row["seq"]}.json'
                if not backup_path.is_file():
                    atomic_json(backup_path, dict(sqlite=sql.snapshot(pid), graph=graph.snapshot(plan)))
                # Replay both writes after any uncertain outcome. Each is idempotent.
                self.event(pid, 'graph_pending')
                graph.write(plan, manifest['batch_id'])
                self.event(pid, 'graph_done')
                publish_image(plan, self.folder, sql.path)
                sql.write(plan)
                self.event(pid, 'completed')
                if number % 25 == 0 or number == len(rows):
                    print(encoded(dict(batch_id=manifest['batch_id'], progress=number, remaining=len(rows)-number)), flush=True)
            except Exception as exc:
                self.event(pid, 'failed', f'{type(exc).__name__}: {exc}')
                raise
        return self.counts()

    def close(self):
        self.conn.close()


def source_manifest(paths):
    return [dict(path=str(Path(p).resolve()), sha256=file_hash(p), bytes=Path(p).stat().st_size) for p in paths]


def identity_index(conn):
    conn.create_function('import_norm', 1, norm, deterministic=True)
    conn.execute('CREATE TEMP TABLE import_titles AS SELECT paper_id,import_norm(title) AS title FROM papers')
    conn.execute('CREATE INDEX temp.import_title_idx ON import_titles(title)')


def check_identity(conn, record, seen_titles):
    pid = record['pid']
    title = next(n['props']['title'] for n in record['nodes'] if n['id'] == pid)
    key = norm(title)
    exists = conn.execute('SELECT 1 FROM papers WHERE paper_id=?', (pid,)).fetchone()
    old = None if exists else conn.execute('SELECT paper_id FROM import_titles WHERE title=? AND paper_id!=? LIMIT 1', (key, pid)).fetchone()
    other = old[0] if old else seen_titles.get(key)
    if other and other != pid:
        raise ValueError(f'Title already belongs to {other}; use its stable paper_id instead of creating {pid}')
    seen_titles[key] = pid


def stage(paths, state_dir, db, graph, mode, image_archives=()):
    sources = source_manifest(paths)
    images = source_manifest(image_archives)
    identity = dict(version=VERSION, sources=sources, images=images, db=str(Path(db).resolve()), graph=graph.identity, relation_mode=mode)
    batch_id = digest(identity)[:20]
    folder = Path(state_dir) / batch_id
    if (folder / 'manifest.json').is_file():
        return batch_id, folder
    temporary = Path(state_dir) / ('.' + batch_id + '.staging')
    temporary.mkdir(parents=True, exist_ok=True)
    batch = Batch(temporary)
    lookup = sqlite3.connect(f'{Path(db).resolve().as_uri()}?mode=ro', uri=True)
    lookup.row_factory = sqlite3.Row
    try:
        identity_index(lookup)
        titles = {}
        # A killed staging pass is rebuilt; no target writes happen during staging.
        with batch.conn:
            batch.conn.execute('DELETE FROM jobs')
        for archive, member, data in iter_graphs(paths):
            record = normalize(data, archive, member)
            check_identity(lookup, record, titles)
            old = batch.conn.execute('SELECT record FROM jobs WHERE pid=?', (record['pid'],)).fetchone()
            if old:
                prev = json.loads(old['record'])
                if prev['nodes'] != record['nodes'] or prev['edges'] != record['edges']:
                    raise ValueError(f'Conflicting snapshots for the same paper_id: {record["pid"]}')
                continue
            previous = lookup.execute('SELECT * FROM papers WHERE paper_id=?', (record['pid'],)).fetchone()
            plan = make_plan(record, dict(previous) if previous else None, graph, mode)
            batch.conn.execute('INSERT INTO jobs(pid,record,plan) VALUES (?,?,?)', (record['pid'], encoded(record), encoded(plan)))
        batch.conn.commit()
        if not batch.conn.execute('SELECT count(*) FROM jobs').fetchone()[0]:
            raise ValueError('No paper graphs found')
        # Check for source replacement between hashing and parsing.
        if source_manifest(paths) != sources:
            raise RuntimeError('Input files changed while staging; retry with stable input files')
        stage_images(temporary, image_archives)
        if source_manifest(image_archives) != images:
            raise RuntimeError('Image archives changed while staging')
        atomic_json(temporary / 'manifest.json', dict(identity, batch_id=batch_id, created_at=stamp()))
    finally:
        lookup.close()
        batch.close()
    temporary.rename(folder)
    return batch_id, folder


def load_settings(path):
    settings = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if sep:
                parts = shlex.split(value, comments=True)
                settings[key.strip()] = parts[0] if parts else ''
    settings.update(os.environ)
    return settings


def preview(paths, db, mode='replace'):
    conn = sqlite3.connect(f'{Path(db).resolve().as_uri()}?mode=ro', uri=True)
    stats = Counter()
    seen = {}
    try:
        identity_index(conn)
        titles = {}
        for archive, member, data in iter_graphs(paths):
            record = normalize(data, archive, member)
            check_identity(conn, record, titles)
            content = digest([record['nodes'], record['edges']])
            if record['pid'] in seen:
                if seen[record['pid']] != content:
                    raise ValueError(f'Conflicting snapshots: {record["pid"]}')
                stats['duplicate_snapshot'] += 1
                continue
            seen[record['pid']] = content
            exists = conn.execute('SELECT 1 FROM papers WHERE paper_id=?', (record['pid'],)).fetchone()
            # Validate metadata in previews too; author resolution is read-only
            # and is deferred to the actual staging pass.
            class PreviewGraph:
                def authors(self, names):
                    return []
            conn.row_factory = sqlite3.Row
            previous = conn.execute('SELECT * FROM papers WHERE paper_id=?', (record['pid'],)).fetchone()
            make_plan(record, dict(previous) if previous else None, PreviewGraph(), mode)
            stats['existing_papers_to_update' if exists else 'new_to_sqlite'] += 1
    finally:
        conn.close()
    if not seen:
        raise ValueError('No paper graphs found')
    return dict(stats, validated_papers=len(seen), writes=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', help='Canonical ZIP/JSON/JSONL/RAR; default: ../new/canonical_*.zip and ECCV2026.rar')
    parser.add_argument('--db', type=Path, default=ROOT / 'papers_fts.db')
    parser.add_argument('--env-file', type=Path, default=ROOT / 'deployment/paperddl.env.bm25')
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'data/incremental_imports')
    parser.add_argument('--image-archive', type=Path, action='append', default=[], help='Optional primary-image ZIP; repeat for multiple ZIPs')
    parser.add_argument('--relation-mode', choices=['replace', 'merge'], default='replace',
                        help='replace: complete per-paper relationship snapshot; merge: only add relationships')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--apply', action='store_true')
    actions.add_argument('--dry-run', action='store_true', help='Default; validates structure and reports SQLite additions/updates')
    parser.add_argument('--resume', help='Resume a staged batch using saved snapshots, without original archives')
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error('SQLite database must already exist; initialize a new index with build_demo_sqlite.py or PaperSearchIndex.build')
    if args.resume and (args.sources or args.image_archive or args.dry_run or not args.apply):
        parser.error('--resume requires --apply and cannot be combined with sources or --dry-run')
    if args.resume and not re.fullmatch('[a-f0-9]{20}', args.resume):
        parser.error('Invalid batch ID')
    paths = [Path(p).resolve() for p in args.sources]
    if not paths and not args.resume:
        paths = sorted((ROOT.parent / 'new').glob('canonical_*.zip'))
        rar = ROOT.parent / 'new/ECCV2026.rar'
        if rar.is_file():
            paths.append(rar)
    if not args.apply:
        print(encoded(preview(paths, args.db, args.relation_mode)))
        return 0
    settings = load_settings(args.env_file)
    if not settings.get('NEO4J_PASSWORD'):
        parser.error('NEO4J_PASSWORD must be set in the environment or --env-file')
    graph = GraphStore(settings.get('NEO4J_HTTP_URI') or 'http://127.0.0.1:17474',
                       settings.get('NEO4J_USER', 'neo4j'), settings['NEO4J_PASSWORD'],
                       settings.get('NEO4J_DATABASE') or 'neo4j')
    graph.preflight()
    with import_lock(args.db):
        args.state_dir.mkdir(parents=True, exist_ok=True)
        expected = args.resume or digest(dict(version=VERSION, sources=source_manifest(paths), images=source_manifest(args.image_archive), db=str(args.db.resolve()),
                                             graph=graph.identity, relation_mode=args.relation_mode))[:20]
        for manifest_path in args.state_dir.glob('*/manifest.json'):
            manifest = json.loads(manifest_path.read_text())
            if manifest['batch_id'] != expected and manifest['db'] == str(args.db.resolve()):
                old = Batch(manifest_path.parent)
                try:
                    if any(k != 'completed' and v for k, v in old.counts().items()):
                        raise RuntimeError(f'Finish the pending batch first: --apply --resume {manifest["batch_id"]}')
                finally:
                    old.close()
        batch_id, folder = (args.resume, args.state_dir / args.resume) if args.resume else stage(paths, args.state_dir, args.db, graph, args.relation_mode, args.image_archive)
        if not (folder / 'manifest.json').is_file():
            raise ValueError('Batch not found')
        print(encoded(dict(batch_id=batch_id, state_dir=str(folder), status='started')), flush=True)
        batch = Batch(folder)
        sql = None
        try:
            sql = SqlStore(args.db)
            counts = batch.run(sql, graph)
            print(encoded(dict(batch_id=batch_id, status='completed', counts=counts, backups=str(folder))))
        except Exception:
            print(encoded(dict(batch_id=batch_id, status='failed', resume=f'--apply --resume {batch_id}', counts=batch.counts())), file=sys.stderr)
            raise
        finally:
            if sql:
                sql.close()
            batch.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
