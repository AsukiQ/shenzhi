import json, sqlite3, requests, time

DB='/root/jiansuo/retrieval_backend/papers_fts.db'
URL='http://127.0.0.1:17474/db/neo4j/query/v2'
AUTH=('neo4j','paperddl_neo4j_2026')
BATCH=1000

conn=sqlite3.connect(DB); conn.execute('PRAGMA busy_timeout=30000')
last=''; matched=0; rows_seen=0
with conn:
  while True:
    escaped=last.replace('\\','\\\\').replace("'","\\'")
    q=f"MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic) WHERE p.paper_id > '{escaped}' RETURN p.paper_id AS pid, collect(DISTINCT t.name) AS topics ORDER BY pid LIMIT {BATCH}"
    last=None
    for attempt in range(3):
      try:
        res=requests.post(URL,auth=AUTH,json={'statement':q},timeout=60); res.raise_for_status(); data=res.json()['data']['values']; break
      except Exception:
        if attempt==2: raise
        time.sleep(2)
    if not data: break
    for pid, topics in data:
      rows_seen += 1
      if not pid: continue
      topics=[str(x).strip() for x in (topics or []) if str(x).strip()]
      if not topics: continue
      row=conn.execute('select 1 from papers where paper_id=?',(pid,)).fetchone()
      if not row: continue
      payload=json.dumps(topics,ensure_ascii=False)
      conn.execute('update papers set subjects_json=? where paper_id=?',(payload,pid))
      conn.execute('delete from papers_fts where paper_id=?',(pid,))
      p=conn.execute('select paper_id,title,abstract,keywords_json from papers where paper_id=?',(pid,)).fetchone()
      kws=' '.join(json.loads(p[3] or '[]')) if p else ''
      conn.execute('insert into papers_fts(paper_id,title,abstract,keywords,subjects) values(?,?,?,?,?)',(pid,p[1],p[2],kws,' '.join(topics)))
      conn.execute('delete from paper_subjects where paper_id=?',(pid,))
      for i,t in enumerate(topics): conn.execute('insert or ignore into paper_subjects(paper_id,value,value_key,position) values(?,?,?,?)',(pid,t,t.lower(),i))
      matched += 1
    last=str(data[-1][0] or '')
    if rows_seen % 10000 == 0: print({'neo4j_rows':rows_seen,'sqlite_matched':matched},flush=True)
    if len(data)<BATCH: break
print(json.dumps({'neo4j_rows':rows_seen,'sqlite_matched':matched}))
