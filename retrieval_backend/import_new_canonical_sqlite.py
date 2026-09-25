import glob, json, sqlite3, zipfile
from pathlib import Path

ROOT = Path('/root/jiansuo')
DB = ROOT / 'retrieval_backend/papers_fts.db'
rows = {}
for zp in glob.glob(str(ROOT / 'new/canonical_*.zip')):
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            if not name.endswith('.json'):
                continue
            data = json.loads(z.read(name))
            papers = [n for n in data.get('graph', {}).get('nodes', []) if n.get('label') == 'Paper']
            if not papers:
                continue
            node = papers[0]; props = node.get('properties', {}); pid = str(node.get('id', '')).strip()
            title = str(props.get('title') or '').strip()
            if not pid or not title:
                continue
            abstract = str(props.get('abstract') or '').strip()
            authors = props.get('authors') if isinstance(props.get('authors'), list) else []
            keywords = props.get('keywords') if isinstance(props.get('keywords'), list) else []
            year = props.get('year')
            try: year = int(year) if year not in (None, '') else None
            except (TypeError, ValueError): year = None
            rows[pid] = (pid, props.get('arxiv_id') or props.get('doi'), title, abstract, props.get('venue') or props.get('conference'), year, authors, keywords, [])
conn = sqlite3.connect(DB); conn.execute('PRAGMA busy_timeout=30000')
inserted = updated = 0
with conn:
    for pid, source, title, abstract, conf, year, authors, keywords, subjects in rows.values():
        exists = conn.execute('select 1 from papers where paper_id=?', (pid,)).fetchone()
        conn.execute('INSERT OR REPLACE INTO papers (paper_id,source_id,title,abstract,conference,year,authors_json,keywords_json,subjects_json) VALUES (?,?,?,?,?,?,?,?,?)', (pid, source, title, abstract, conf, year, json.dumps(authors, ensure_ascii=False), json.dumps(keywords, ensure_ascii=False), json.dumps(subjects, ensure_ascii=False)))
        conn.execute('delete from papers_fts where paper_id=?', (pid,))
        conn.execute('insert into papers_fts(paper_id,title,abstract,keywords,subjects) values(?,?,?,?,?)', (pid, title, abstract, ' '.join(map(str, keywords)), ' '.join(map(str, subjects))))
        for table, vals in [('paper_authors', authors), ('paper_keywords', keywords), ('paper_subjects', subjects)]:
            conn.execute(f'delete from {table} where paper_id=?', (pid,))
            for i, value in enumerate(vals):
                text = str(value); conn.execute(f'insert or ignore into {table}(paper_id,value,value_key,position) values(?,?,?,?)', (pid, text, text.strip().lower(), i))
        inserted += not bool(exists); updated += bool(exists)
print(json.dumps({'canonical_papers': len(rows), 'inserted': inserted, 'updated': updated}, ensure_ascii=False))
