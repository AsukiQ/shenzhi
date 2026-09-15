"""Memory-bounded merge of the authoritative rich graph and legacy CSV ZIP."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,re,shutil,sqlite3,sys,tempfile,zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator,Sequence
from .build_rich_graph import EDGE_CONTRACT,EDGE_FILES,NODE_FILES,NODE_LABELS,SCHEMA,_schema_cypher,_import_script
from .prepare import file_sha256,is_placeholder_abstract

def _csv_limit():
    n=sys.maxsize
    while True:
        try: csv.field_size_limit(n); return
        except OverflowError: n//=10
_csv_limit()
PAPER_FIELDS=["paper_id:ID(Paper)","title","abstract","arxiv_id","doi","url","pdf_url","year:int","external:boolean","venue","booktitle","publisher","authors_json","keywords_json","datasets_json","code_resources_json","research_problem","motivation","contributions_json","result","properties_json","sources_json",":LABEL"]
def norm(v): return " ".join((v or "").casefold().split())
def arxiv(v):
    x=norm(v);x=re.sub(r'^https?://arxiv\.org/(?:abs|pdf)/','',x).removesuffix('.pdf');return re.sub(r'v\d+$','',x)
def doi(v): return norm(v).removeprefix('https://doi.org/').removeprefix('doi:').strip()
def title_key(v): return re.sub(r'[^\w\s]',' ',norm(v),flags=re.UNICODE)
def identity_keys(row):
    out=[]
    if arxiv(row.get('arxiv_id','')): out.append(('arxiv',arxiv(row['arxiv_id'])))
    if doi(row.get('doi','')): out.append(('doi',doi(row['doi'])))
    if title_key(row.get('title','')): out.append(('title',title_key(row['title'])))
    return out
def sid(p,*v):
    """Return a deterministic ID for a legacy entity.

    Keep the separator outside the f-string expression: Python rejects
    backslash escapes inside an f-string expression on supported runtimes.
    """
    material = "\x1f".join(v).encode()
    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"{p}:legacy:{digest}"
def jdict(v):
    try: x=json.loads(v or '{}'); return x if isinstance(x,dict) else {}
    except json.JSONDecodeError:return {}
def jlist(v):
    try: x=json.loads(v or '[]'); return x if isinstance(x,list) else []
    except json.JSONDecodeError:return []
def member(z,b):
    x=[n for n in z.namelist() if n.endswith('/paper_kg/'+b+'.csv')]
    if len(x)!=1: raise ValueError(f'expected one legacy {b}.csv, got {x}')
    return x[0]
def zrows(z,b)->Iterator[dict[str,str]]:
    with z.open(member(z,b)) as raw:
        with io.TextIOWrapper(raw,encoding='utf-8',newline='') as f: yield from csv.DictReader(f)
def frows(p):
    h=p.open(encoding='utf-8',newline=''); r=csv.DictReader(h); fields=list(r.fieldnames or [])
    def it():
        try: yield from r
        finally: h.close()
    return fields,it()
def writer(p,fields,append=False):
    h=p.open('a' if append else 'w',encoding='utf-8',newline=''); w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n',extrasaction='ignore')
    if not append:w.writeheader()
    return h,w
def legacy(row):
    props={k:row.get(k,'') for k in ('dblp_key','biburl','type','source','source_file') if row.get(k)}
    return {'paper_id:ID(Paper)':row.get('paper_id:ID(Paper)',''),'title':row.get('title',''),'abstract':row.get('abstract',''),'arxiv_id':row.get('arxiv_id',''),'doi':row.get('doi',''),'url':row.get('biburl',''),'pdf_url':row.get('pdf_url',''),'year:int':row.get('year:int',''),'external:boolean':'false','venue':row.get('venue',''),'booktitle':'','publisher':'','authors_json':'[]','keywords_json':'[]','datasets_json':'[]','code_resources_json':'[]','research_problem':'','motivation':'','contributions_json':'[]','result':'','properties_json':json.dumps(props,sort_keys=True),'sources_json':'{"legacy_candidate": true}',':LABEL':'Paper'}
def mergepaper(new,old):
    out=dict(new); oldx=legacy(old); filled=0
    for f in ('title','abstract','arxiv_id','doi','url','pdf_url','year:int','venue','booktitle','publisher'):
        if not str(out.get(f,'')).strip() and str(oldx.get(f,'')).strip():out[f]=oldx[f];filled+=1
    if norm(str(out.get('external:boolean',''))) in {'true','1','yes'} and old.get('title') and not is_placeholder_abstract(old.get('abstract','')):out['external:boolean']='false'
    p=jdict(out.get('properties_json',''));p.setdefault('legacy_fill',jdict(oldx['properties_json']));out['properties_json']=json.dumps(p,sort_keys=True)
    s=jdict(out.get('sources_json',''));s['legacy_candidate_fill']=True;out['sources_json']=json.dumps(s,sort_keys=True)
    return out,filled
def db(path):
    c=sqlite3.connect(path);c.execute('PRAGMA journal_mode=OFF');c.execute('PRAGMA synchronous=OFF');c.executescript('CREATE TABLE nodes(id TEXT PRIMARY KEY,label TEXT);CREATE TABLE papers(id TEXT PRIMARY KEY,row TEXT,newmain INT);CREATE TABLE old(id TEXT PRIMARY KEY,row TEXT);CREATE TABLE paper_map(old_id TEXT PRIMARY KEY,target_id TEXT);CREATE TABLE identities(kind TEXT,key TEXT,paper_id TEXT,PRIMARY KEY(kind,key,paper_id));CREATE INDEX identities_lookup ON identities(kind,key);CREATE TABLE map(kind TEXT,old TEXT,target TEXT,PRIMARY KEY(kind,old));CREATE TABLE names(label TEXT,norm TEXT,id TEXT,PRIMARY KEY(label,norm,id));CREATE INDEX names_by_id ON names(id);CREATE TABLE edges(t TEXT,s TEXT,e TEXT,PRIMARY KEY(t,s,e));CREATE TABLE years(id TEXT PRIMARY KEY,y TEXT);CREATE TABLE pv(p TEXT PRIMARY KEY,v TEXT);CREATE TABLE py(p TEXT PRIMARY KEY,y TEXT);CREATE TABLE conf(p TEXT PRIMARY KEY,c TEXT,v TEXT);');return c
def paper_target(c,p):
    x=c.execute('SELECT target_id FROM paper_map WHERE old_id=?',(p,)).fetchone();return x[0] if x else p
def addnode(c,label,row,field):
    i=row[field];c.execute('INSERT OR IGNORE INTO nodes VALUES(?,?)',(i,label));c.execute('INSERT OR IGNORE INTO names VALUES(?,?,?)',(label,norm(row.get('name','')),i))
def merge_rich_legacy(rich_dir,legacy_zip,output_dir):
    rich_dir,legacy_zip,output_dir=map(lambda x:Path(x).resolve(),(rich_dir,legacy_zip,output_dir))
    if output_dir.exists():raise FileExistsError(output_dir)
    if not (rich_dir/'validation_report.json').is_file() or not legacy_zip.is_file():raise FileNotFoundError('rich validation report and legacy ZIP are required')
    if not json.loads((rich_dir/'validation_report.json').read_text()).get('ready'):raise ValueError('rich candidate validation is not ready')
    temp=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'.tmp-',dir=output_dir.parent));kg=temp/'paper_kg';kg.mkdir();c=db(temp.parent/(temp.name+'.sqlite'));q=Counter();conf=Counter();pc=Counter()
    try:
      with zipfile.ZipFile(legacy_zip) as z:
        for old in zrows(z,'papers'):
          i=old.get('paper_id:ID(Paper)','').strip()
          if not i: q['blank_legacy_paper_id']+=1;continue
          if c.execute('INSERT OR IGNORE INTO old VALUES(?,?)',(i,json.dumps(old))).rowcount:pc['legacy_papers']+=1
        fields,rows=frows(rich_dir/'paper_kg'/'papers.csv');
        if fields!=PAPER_FIELDS:raise ValueError('unexpected rich papers.csv contract')
        for row in rows:
          i=row['paper_id:ID(Paper)'];main=int(norm(str(row.get('external:boolean',''))) not in {'true','1','yes'});c.execute('INSERT OR REPLACE INTO papers VALUES(?,?,?)',(i,json.dumps(row),main));
          for kind,key in identity_keys(row): c.execute('INSERT OR IGNORE INTO identities VALUES(?,?,?)',(kind,key,i))
          pc['new_papers']+=1
        for old_id,encoded in c.execute('SELECT id,row FROM old').fetchall():
          old=json.loads(encoded); candidates=[]; conflict_keys=[]
          for kind,key in identity_keys(old):
            ids=[x[0] for x in c.execute('SELECT paper_id FROM identities WHERE kind=? AND key=?',(kind,key)).fetchall()]
            if len(ids)==1: candidates.append((kind,ids[0]))
            elif len(ids)>1: conflict_keys.append(kind)
          unique={target for _,target in candidates}
          if conflict_keys or len(unique)>1:
            q['identity_conflicts']+=1;c.execute('INSERT OR IGNORE INTO paper_map VALUES(?,?)',(old_id,old_id));c.execute('INSERT OR IGNORE INTO papers VALUES(?,?,0)',(old_id,json.dumps(legacy(old))));c.execute("INSERT OR IGNORE INTO nodes VALUES(?, 'Paper')",(old_id,));pc['quarantined_legacy_papers']+=1;continue
          if len(unique)==1:
            target=next(iter(unique));c.execute('INSERT OR REPLACE INTO paper_map VALUES(?,?)',(old_id,target));current=c.execute('SELECT row FROM papers WHERE id=?',(target,)).fetchone();merged,n=mergepaper(json.loads(current[0]),old);c.execute('UPDATE papers SET row=? WHERE id=?',(json.dumps(merged),target));pc['identity_matched_papers']+=1
            matched_kinds={kind for kind,candidate in candidates if candidate==target}
            for kind in matched_kinds: pc[f'identity_matched_by_{kind}']+=1
            conf['legacy_fields_filled']+=n
          else:
            c.execute('INSERT OR IGNORE INTO paper_map VALUES(?,?)',(old_id,old_id));c.execute('INSERT INTO papers VALUES(?,?,?)',(old_id,json.dumps(legacy(old)),0));c.execute("INSERT OR IGNORE INTO nodes VALUES(?, 'Paper')",(old_id,));pc['legacy_only_papers']+=1
        pc['merged_papers']=c.execute('SELECT count(*) FROM papers').fetchone()[0]
        c.execute("INSERT OR IGNORE INTO nodes SELECT id,'Paper' FROM papers")
        hp,wp=writer(kg/'papers.csv',PAPER_FIELDS)
        for (r,) in c.execute('SELECT row FROM papers ORDER BY newmain DESC,id'):wp.writerow(json.loads(r))
        hp.close()
        fby={}
        for label in NODE_LABELS[1:]:
          fields,rows=frows(rich_dir/'paper_kg'/NODE_FILES[label]);fby[label]=fields;h,w=writer(kg/NODE_FILES[label],fields)
          idf=next(x for x in fields if ':ID(' in x)
          for row in rows:w.writerow(row);addnode(c,label,row,idf)
          h.close()
        for oldname,label,prefix in (('authors','Author','author'),('keywords','Topic','topic'),('subjects','Topic','topic'),('venues','Venue','venue')):
          h,w=writer(kg/NODE_FILES[label],fby[label],append=True);idf=next(x for x in fby[label] if ':ID(' in x)
          for row in zrows(z,oldname):
            oid=next((row[k] for k in row if ':ID(' in k),'');name=row.get('name','');ms=c.execute('SELECT id FROM names WHERE label=? AND norm=?',(label,norm(name))).fetchall();target=ms[0][0] if len(ms)==1 else sid(prefix,oid,name);c.execute('INSERT OR IGNORE INTO map VALUES(?,?,?)',(oldname,oid,target))
            if not c.execute('SELECT 1 FROM nodes WHERE id=?',(target,)).fetchone():w.writerow({idf:target,'name':name,'properties_json':json.dumps({'legacy_id':oid}),'sources_json':'{"legacy_candidate": true}',':LABEL':label});addnode(c,label,{idf:target,'name':name},idf)
          h.close()
        for row in zrows(z,'years'):c.execute('INSERT OR IGNORE INTO years VALUES(?,?)',(row.get('year_id:ID(Year)',''),row.get('year:int','')))
        for row in zrows(z,'published_in'):c.execute('INSERT OR IGNORE INTO pv VALUES(?,?)',(row.get(':START_ID(Paper)',''),row.get(':END_ID(Venue)','')))
        for row in zrows(z,'published_year'):
          y=c.execute('SELECT y FROM years WHERE id=?',(row.get(':END_ID(Year)',''),)).fetchone();c.execute('INSERT OR IGNORE INTO py VALUES(?,?)',(row.get(':START_ID(Paper)',''),y[0] if y else ''))
        h,w=writer(kg/NODE_FILES['Conference'],fby['Conference'],append=True)
        for pid,vid in c.execute('SELECT pv.p,pv.v FROM pv').fetchall():
          target=c.execute("SELECT target FROM map WHERE kind='venues' AND old=?",(vid,)).fetchone();
          if not target:continue
          vn=c.execute('SELECT id FROM names WHERE label="Venue" AND id=?',(target[0],)).fetchone();name=c.execute('SELECT norm FROM names WHERE id=?',(target[0],)).fetchone()[0] if vn else '';year=c.execute('SELECT y FROM py WHERE p=?',(pid,)).fetchone();year=year[0] if year else '';cid=sid('conference',target[0],year);c.execute('INSERT OR IGNORE INTO conf VALUES(?,?,?)',(pid,cid,target[0]));
          if not c.execute('SELECT 1 FROM nodes WHERE id=?',(cid,)).fetchone():w.writerow({'conference_id:ID(Conference)':cid,'name':(name+' '+year).strip(),'venue':name,'year:int':year,'booktitle':'','publisher':'','properties_json':'{"legacy": true}','sources_json':'{"legacy_candidate": true}',':LABEL':'Conference'});addnode(c,'Conference',{'conference_id:ID(Conference)':cid,'name':name},'conference_id:ID(Conference)')
        h.close()
        ec=Counter()
        for et in EDGE_CONTRACT:
          fields,rows=frows(rich_dir/'paper_kg'/EDGE_FILES[et]);h,w=writer(kg/EDGE_FILES[et],fields);sf=next(x for x in fields if ':START_ID(' in x);ef=next(x for x in fields if ':END_ID(' in x)
          for row in rows:w.writerow(row);c.execute('INSERT OR IGNORE INTO edges VALUES(?,?,?)',(et,row[sf],row[ef]));ec[et]+=1
          def add(s,t,extra=None,category='legacy_only'):
            if not s or not t or not c.execute('SELECT 1 FROM nodes WHERE id=?',(s,)).fetchone() or not c.execute('SELECT 1 FROM nodes WHERE id=?',(t,)).fetchone():q[f'{et}_{category}_rejected']+=1;return
            if not c.execute('INSERT OR IGNORE INTO edges VALUES(?,?,?)',(et,s,t)).rowcount:q[f'{et}_{category}_deduplicated']+=1;return
            x={sf:s,ef:t,':TYPE':et,'properties_json':'{}','sources_json':'{"legacy_candidate": true}','derived:boolean':'true'};x.update(extra or {});w.writerow(x);ec[et]+=1
            q[f'{et}_{category}_migrated']+=1
          if et=='AUTHORED_BY':
            for x in zrows(z,'authored_by'):
              state=c.execute('SELECT newmain FROM papers WHERE id=?',(x.get(':START_ID(Paper)',''),)).fetchone()
              if state and state[0]:continue
              oldp=x.get(':START_ID(Paper)','');cat='identity_matched' if paper_target(c,oldp)!=oldp else 'legacy_only';t=c.execute("SELECT target FROM map WHERE kind='authors' AND old=?",(x.get(':END_ID(Author)',''),)).fetchone();add(paper_target(c,oldp),t[0] if t else '',{'position:int':x.get('author_order:int','')},cat)
          elif et=='HAS_TOPIC':
            for fn,key,kind in (('has_keyword',':END_ID(Keyword)','keywords'),('has_subject',':END_ID(Subject)','subjects')):
              for x in zrows(z,fn):
                p=x.get(':START_ID(Paper)','');m=c.execute('SELECT newmain FROM papers WHERE id=?',(p,)).fetchone();t=c.execute('SELECT target FROM map WHERE kind=? AND old=?',(kind,x.get(key,''))).fetchone();
                if not (m and m[0]):add(paper_target(c,p),t[0] if t else '',category='identity_matched' if paper_target(c,p)!=p else 'legacy_only')
          elif et=='PUBLISHED_IN':
            for p,cid,v in c.execute('SELECT p,c,v FROM conf').fetchall():
              m=c.execute('SELECT newmain FROM papers WHERE id=?',(p,)).fetchone();
              if not (m and m[0]):add(paper_target(c,p),cid,category='identity_matched' if paper_target(c,p)!=p else 'legacy_only')
          elif et=='PART_OF':
            for _,cid,v in c.execute('SELECT p,c,v FROM conf').fetchall():add(cid,v)
          elif et=='CITES':
            for x in zrows(z,'cites'):
              p=x.get(':START_ID(Paper)','');m=c.execute('SELECT newmain FROM papers WHERE id=?',(p,)).fetchone();
              if not (m and m[0]):add(paper_target(c,p),paper_target(c,x.get(':END_ID(Paper)','')) ,{'source':x.get('source','legacy')},'identity_matched' if paper_target(c,p)!=p else 'legacy_only')
          h.close()
      docs=0
      with (temp/'retrieval_documents.jsonl').open('w',encoding='utf-8',newline='\n') as out:
        for (r,) in c.execute('SELECT row FROM papers ORDER BY newmain DESC,id'):
          x=json.loads(r)
          if not x.get('title') or is_placeholder_abstract(str(x.get('abstract',''))) or norm(str(x.get('external:boolean',''))) in {'true','1','yes'}:continue
          d={'paper_id':x['paper_id:ID(Paper)'],'title':x.get('title',''),'abstract':x.get('abstract',''),'year':x.get('year:int',''),'conference':x.get('venue',''),'authors':jlist(x.get('authors_json','[]')),'keywords':jlist(x.get('keywords_json','[]'))};d['search_text']='\n'.join(str(d[k]) for k in ('title','abstract','conference'));out.write(json.dumps(d,ensure_ascii=False,sort_keys=True)+'\n');docs+=1
      (temp/'constraints.cypher').write_text(_schema_cypher(),encoding='utf-8');s=temp/'import_neo4j_admin.sh';s.write_text(_import_script(),encoding='utf-8');s.chmod(0o755)
      qpath=temp/'merge_quarantine.json';qpath.write_text(json.dumps(dict(q),indent=2,sort_keys=True),encoding='utf-8')
      nodes=Counter(dict(c.execute('SELECT label,count(*) FROM nodes GROUP BY label')));quality={'schema':SCHEMA,'merge_policy':'new_authoritative_legacy_fill_v1','nodes_by_label':dict(nodes),'edges_by_type':dict(ec),'counts':{**dict(pc),'retrieval_documents':docs},'conflict_counts':dict(conf),'quarantine_counts':dict(q),'quality_gates':{'new_graph_authoritative':True,'legacy_only_fills_missing_or_historical':True,'disk_backed_streaming_merge':True}}
      (temp/'quality_report.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');outputs={}
      for p in sorted(x for x in temp.rglob('*') if x.is_file() and x.name not in {'manifest.json','quality_report.json'}):outputs[p.relative_to(temp).as_posix()]={'bytes':p.stat().st_size,'sha256':file_sha256(p)}
      manifest={'schema':SCHEMA,'merge_policy':quality['merge_policy'],'inputs':[{'path':str(rich_dir/'manifest.json'),'bytes':(rich_dir/'manifest.json').stat().st_size,'sha256':file_sha256(rich_dir/'manifest.json')},{'path':str(legacy_zip),'bytes':legacy_zip.stat().st_size,'sha256':file_sha256(legacy_zip)}],'outputs':outputs,'counts':quality['counts'],'conflict_counts':quality['conflict_counts'],'quarantine_counts':quality['quarantine_counts']};(temp/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');c.close();(temp.parent/(temp.name+'.sqlite')).unlink(missing_ok=True);temp.replace(output_dir);return quality
    except Exception:
      c.close();(temp.parent/(temp.name+'.sqlite')).unlink(missing_ok=True);shutil.rmtree(temp,ignore_errors=True);raise
def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rich-dir',required=True,type=Path);p.add_argument('--legacy-zip',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args(argv);print(json.dumps(merge_rich_legacy(a.rich_dir,a.legacy_zip,a.output_dir),ensure_ascii=False,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
