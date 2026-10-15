"""Failure recovery and graph/FTS integration tests; live tests use unique test IDs."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid
import zipfile

from import_new_batch import (Batch, GraphStore, SqlStore, import_lock, load_settings, main,
                              make_plan, normalize, preview, stage)
from paper_search import PaperSearchIndex, SearchFilters


def snapshot(pid='paper:test-main', topic='Uniqueoldtopic', abstract='Uniqueoldterm', cite='paper:test-reference'):
    nodes = [dict(id=pid, label='Paper', properties={'title': 'Incremental import research paper', 'abstract': abstract}),
             dict(id='topic:' + topic, label='Topic', properties={'name': topic})]
    edges = [dict(source=pid, target='topic:' + topic, type='HAS_TOPIC')]
    if cite:
        nodes.append(dict(id=cite, label='Paper', properties={'title': 'A reference research publication'}))
        edges.append(dict(source=pid, target=cite, type='CITES'))
    return {'main_paper_id': pid, 'graph': {'nodes': nodes, 'edges': edges}}


class RecordingGraph:
    identity = 'test://graph'

    def __init__(self):
        self.writes = []
        self.fail_after_commit = False

    def authors(self, names):
        return []

    def snapshot(self, plan):
        return []

    def write(self, plan, batch):
        self.writes.append(plan['revision'])
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise TimeoutError('Simulated lost response after graph commit')


class IncrementalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'backend' / 'test.db'
        self.db.parent.mkdir()
        self.index = PaperSearchIndex(self.db)
        conn = self.index.connect(writable=True)
        conn.close()
        self.addCleanup(self.index.close)
        self.graph = RecordingGraph()
        self.state = self.root / 'batches'
        self.state.mkdir()

    def plan(self, data):
        record = normalize(data, 'batch.zip', 'canonical_CONF/paper_canonical.json')
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            previous = conn.execute('SELECT * FROM papers WHERE paper_id=?', (record['pid'],)).fetchone()
        return make_plan(record, dict(previous) if previous else None, self.graph, 'replace')

    def stage(self, data, images=()):
        path = self.root / 'input.json'
        path.write_text(json.dumps(data))
        return stage([path], self.state, self.db, self.graph, 'replace', images)

    def test_online_update_replaces_fts_and_relations_without_restarting_reader(self):
        sql = SqlStore(self.db)
        self.addCleanup(sql.close)
        sql.write(self.plan(snapshot()))
        self.assertEqual(len(self.index.recall('Uniqueoldterm')), 1)
        row_id = sql.previous('paper:test-main')['row_id']
        sql.write(self.plan(snapshot(topic='Uniquenewtopic', abstract='Uniquenewterm', cite=None)))
        self.assertEqual(self.index.recall('Uniqueoldterm'), [])
        self.assertEqual(len(self.index.recall('Uniquenewterm')), 1)
        self.assertEqual(self.index.search('', filters=SearchFilters(subject=['Uniqueoldtopic']))[0], [])
        self.assertEqual(sql.previous('paper:test-main')['row_id'], row_id)
        self.assertEqual(sql.conn.execute('SELECT count(*) FROM papers_fts').fetchone()[0], 1)
        self.assertEqual(sql.conn.execute('SELECT value FROM paper_subjects').fetchone()[0], 'Uniquenewtopic')

    def test_legacy_duplicate_fts_rows_are_removed(self):
        with sqlite3.connect(self.db) as conn:
            conn.executemany('INSERT INTO papers_fts(paper_id,title,abstract) VALUES (?,?,?)',
                             [('paper:test-main', 'legacy', 'Legacytoken')] * 3)
        sql = SqlStore(self.db)
        try:
            plan = self.plan(snapshot())
            sql.write(plan)
            sql.write(plan)
            self.assertEqual(sql.conn.execute('SELECT count(*) FROM papers_fts').fetchone()[0], 1)
            self.assertEqual(self.index.recall('Legacytoken'), [])
        finally:
            sql.close()

    def test_resume_after_graph_success_and_sqlite_failure_without_original_input(self):
        batch_id, folder = self.stage(snapshot())
        batch = Batch(folder)
        sql = SqlStore(self.db)
        sql.write = lambda plan: (_ for _ in ()).throw(sqlite3.OperationalError('Simulated SQLite failure'))
        try:
            with self.assertRaises(sqlite3.OperationalError):
                batch.run(sql, self.graph)
            self.assertEqual(batch.counts(), {'failed': 1})
            self.assertIsNone(sql.previous('paper:test-main'))
            self.assertTrue((folder / 'before_1.json').is_file())
        finally:
            sql.close()
            batch.close()
        (self.root / 'input.json').unlink()
        batch = Batch(folder)
        sql = SqlStore(self.db)
        try:
            self.assertEqual(batch.run(sql, self.graph), {'completed': 1})
            self.assertEqual(batch.run(sql, self.graph), {'completed': 1})
            self.assertEqual(len(self.graph.writes), 2)
            self.assertEqual(sql.conn.execute('SELECT count(*) FROM papers_fts').fetchone()[0], 1)
        finally:
            sql.close()
            batch.close()

    def test_resume_after_lost_graph_commit_response(self):
        _, folder = self.stage(snapshot())
        batch, sql = Batch(folder), SqlStore(self.db)
        self.graph.fail_after_commit = True
        try:
            with self.assertRaises(TimeoutError):
                batch.run(sql, self.graph)
            self.assertEqual(batch.run(sql, self.graph), {'completed': 1})
            self.assertEqual(len(set(self.graph.writes)), 1)
        finally:
            sql.close()
            batch.close()

    def test_resume_after_sqlite_commit_before_journal_completion(self):
        _, folder = self.stage(snapshot())
        batch, sql = Batch(folder), SqlStore(self.db)
        event = batch.event
        try:
            def fail_completion(pid, stage, error=None):
                if stage == 'completed':
                    raise RuntimeError('Simulated process exit between commits')
                event(pid, stage, error)
            batch.event = fail_completion
            with self.assertRaises(RuntimeError):
                batch.run(sql, self.graph)
            batch.event = event
            self.assertEqual(batch.run(sql, self.graph), {'completed': 1})
            self.assertEqual(sql.conn.execute('SELECT count(*) FROM papers_fts').fetchone()[0], 1)
        finally:
            sql.close()
            batch.close()

    def test_preview_is_read_only_and_rejects_ambiguous_or_unsafe_input(self):
        source = self.root / 'preview.json'
        source.write_text(json.dumps(snapshot()))
        self.assertEqual(preview([source], self.db)['new_to_sqlite'], 1)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM papers').fetchone()[0], 0)
        bad = snapshot()
        bad['graph']['edges'][0]['type'] = 'BAD` DELETE n'
        with self.assertRaises(ValueError):
            normalize(bad, 'source', 'member')
        bad = snapshot()
        del bad['main_paper_id']
        bad['graph']['edges'] = []
        with self.assertRaises(ValueError):
            normalize(bad, 'source', 'member')

    def test_unchanged_batch_is_reused_and_conflicting_paper_id_is_rejected(self):
        first = self.stage(snapshot())
        self.assertEqual(first, self.stage(snapshot()))
        sql = SqlStore(self.db)
        try:
            sql.write(self.plan(snapshot()))
        finally:
            sql.close()
        with self.assertRaisesRegex(ValueError, 'stable paper_id'):
            self.stage(snapshot(pid='paper:accidental-duplicate'))

    def test_only_one_importer_can_write(self):
        with import_lock(self.db):
            with self.assertRaisesRegex(RuntimeError, 'Another incremental import'):
                with import_lock(self.db):
                    pass

    def test_invalid_metadata_is_rejected_before_graph_writes(self):
        data = snapshot()
        data['graph']['nodes'][0]['properties']['year'] = 'not-a-year'
        with self.assertRaises(ValueError):
            self.stage(data)
        self.assertEqual(self.graph.writes, [])

    def test_images_are_staged_and_published_and_manifest_reloads(self):
        source = self.root / 'graphs.zip'
        with zipfile.ZipFile(source, 'w') as arc:
            arc.writestr('canonical_CONF/paper_canonical.json', json.dumps(snapshot()))
        images = self.root / 'images.zip'
        with zipfile.ZipFile(images, 'w') as arc:
            arc.writestr('CONF/paper_minimal.png', b'example-image-bytes')
        # Reader already exists before the import.
        self.assertIsNone(self.index.image_id_for('paper:test-main'))
        _, folder = stage([source], self.state, self.db, self.graph, 'replace', [images])
        source.unlink()
        images.unlink()
        batch, sql = Batch(folder), SqlStore(self.db)
        try:
            batch.run(sql, self.graph)
            path = self.index.image_id_for('paper:test-main')
            self.assertTrue(path.startswith('/images/incremental/'))
            self.assertEqual((self.db.parent.parent / path.lstrip('/')).read_bytes(), b'example-image-bytes')
        finally:
            sql.close()
            batch.close()


@unittest.skipUnless(os.environ.get('LIVE_INCREMENTAL_TEST') == '1', 'Set LIVE_INCREMENTAL_TEST=1 for Neo4j integration')
class LiveGraphTest(unittest.TestCase):
    def test_promotion_snapshot_replacement_incoming_citations_and_reference_protection(self):
        settings = load_settings(Path(__file__).with_name('deployment') / 'paperddl.env.bm25')
        graph = GraphStore(settings.get('NEO4J_HTTP_URI') or 'http://127.0.0.1:17474',
                           settings.get('NEO4J_USER', 'neo4j'), settings['NEO4J_PASSWORD'])
        prefix = '__incremental_test_' + uuid.uuid4().hex + ':'
        main, incoming, reference = [prefix + k for k in ('main', 'incoming', 'reference')]
        try:
            graph.preflight()
            graph.query('''CREATE (p:Paper {paper_id:$main,external:true}),
                (i:Paper {paper_id:$incoming,external:false}),
                (r:Paper {paper_id:$reference,external:false,title:'Preserve full reference title',abstract:'Full reference data'})
                CREATE (i)-[:CITES]->(p)''', dict(main=main, incoming=incoming, reference=reference))
            data = snapshot(main, topic=prefix+'oldtopic', cite=reference)
            record = normalize(data, 'test.zip', 'test.json')
            plan = make_plan(record, None, graph, 'replace')
            # snapshot query must also execute successfully against real Neo4j.
            self.assertTrue(graph.snapshot(plan))
            graph.write(plan, 'test')
            graph.write(plan, 'test')
            rows = graph.query('''MATCH (p:Paper {paper_id:$id})
                RETURN p.external AS external, count { (p)-[:HAS_TOPIC]->() } AS topics,
                count { ()-[:CITES]->(p) } AS incoming''', {'id': main})
            self.assertEqual(rows, [dict(external=False, topics=1, incoming=1)])
            self.assertEqual(graph.query('MATCH (p:Paper {paper_id:$id}) RETURN p.title AS title,p.external AS external',
                                         {'id': reference}), [dict(title='Preserve full reference title',external=False)])
            changed = snapshot(main, topic=prefix+'newtopic', cite=None)
            graph.write(make_plan(normalize(changed, 'test.zip', 'test.json'), None, graph, 'replace'), 'test-update')
            rows = graph.query('''MATCH (p:Paper {paper_id:$id})
                RETURN count { (p)-[:CITES]->() } AS outgoing, count { ()-[:CITES]->(p) } AS incoming,
                [(p)-[:HAS_TOPIC]->(t) | t.name] AS topics''', {'id': main})
            self.assertEqual(rows, [dict(outgoing=0, incoming=1, topics=[prefix+'newtopic'])])
            # Invalid second statement must roll back the entire HTTP transaction.
            with self.assertRaises(RuntimeError):
                graph.transaction([('MATCH (p:Paper {paper_id:$id}) SET p.title="should rollback"', {'id': main}),
                                   ('THIS IS INVALID CYPHER', {})])
            self.assertEqual(graph.query('MATCH (p:Paper {paper_id:$id}) RETURN p.title AS title', {'id': main})[0]['title'],
                             'Incremental import research paper')
        finally:
            # Touch only this test's unique Paper and Topic IDs.
            graph.query('MATCH (p:Paper) WHERE p.paper_id STARTS WITH $prefix DETACH DELETE p', {'prefix': prefix})
            graph.query('MATCH (t:Topic) WHERE t.topic_id STARTS WITH $prefix DETACH DELETE t', {'prefix': 'topic:' + prefix})

    def test_cli_import_update_and_completed_batch_replay(self):
        settings = load_settings(Path(__file__).with_name('deployment') / 'paperddl.env.bm25')
        graph = GraphStore(settings.get('NEO4J_HTTP_URI') or 'http://127.0.0.1:17474',
                           settings.get('NEO4J_USER', 'neo4j'), settings['NEO4J_PASSWORD'])
        prefix = '__incremental_test_' + uuid.uuid4().hex + ':'
        pid = prefix + 'main'
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                db = root / 'test.db'
                conn = PaperSearchIndex(db).connect(writable=True)
                conn.close()
                source = root / 'snapshot.json'
                source.write_text(json.dumps(snapshot(pid, prefix+'oldtopic', 'Uniqueoldterm', cite=None)))
                args = [str(source), '--db', str(db), '--state-dir', str(root/'batches'), '--apply']
                self.assertEqual(main(args), 0)
                reader = PaperSearchIndex(db)
                try:
                    self.assertEqual(len(reader.recall('Uniqueoldterm')), 1)
                    source.write_text(json.dumps(snapshot(pid, prefix+'newtopic', 'Uniquenewterm', cite=None)))
                    self.assertEqual(main(args), 0)
                    self.assertEqual(main(args), 0)
                    self.assertEqual(reader.recall('Uniqueoldterm'), [])
                    self.assertEqual(len(reader.recall('Uniquenewterm')), 1)
                    with sqlite3.connect(db) as check:
                        self.assertEqual(check.execute('SELECT count(*) FROM papers_fts').fetchone()[0], 1)
                    self.assertEqual(graph.query('MATCH (p:Paper {paper_id:$id}) RETURN p.abstract AS abstract',
                                                 {'id': pid}), [{'abstract':'Uniquenewterm'}])
                finally:
                    reader.close()
        finally:
            graph.query('MATCH (p:Paper) WHERE p.paper_id STARTS WITH $prefix DETACH DELETE p', {'prefix': prefix})
            graph.query('MATCH (t:Topic) WHERE t.topic_id STARTS WITH $prefix DETACH DELETE t', {'prefix': 'topic:' + prefix})


if __name__ == '__main__':
    unittest.main()
