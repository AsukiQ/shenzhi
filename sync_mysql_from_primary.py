#!/usr/bin/env python3
"""Reconcile MySQL papers with Neo4j primary-paper JSONL. Apply is explicit."""
import argparse,json,os,shlex,subprocess
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--env',default='/home/user2/MKC_Paperddl/server/.env');ap.add_argument('--apply',action='store_true');ap.add_argument('--incremental',action='store_true',help='only insert paper_ids absent from MySQL');ap.add_argument('--report',required=True);args=ap.parse_args()
 rows=[];ids=set();bad=[]
 for line in Path(args.input).open():
  r=json.loads(line);pid=r.get('paper_id','')
  if not pid or pid in ids: bad.append(pid or '<missing>');continue
  ids.add(pid);rows.append(r)
 report={'source_primary_ids':len(ids),'duplicate_input_ids':len(bad),'duplicate_input_samples':bad[:20],'apply':args.apply,'incremental':args.incremental}
 # The actual write is delegated to a Node script using the application's mysql2 config.
 report['mysql_sync_script']='sync-mysql-papers.mjs'
 Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=False))
 if not args.apply:return
 env=os.environ.copy();env['SYNC_INPUT']=str(Path(args.input).resolve());env['SYNC_REPORT']=str(Path(args.report).resolve());env['SYNC_ENV']=str(Path(args.env).resolve());env['SYNC_INCREMENTAL']='1' if args.incremental else '0'
 subprocess.run(['node',str(Path(__file__).with_name('sync-mysql-papers.mjs'))],env=env,check=True)
if __name__=='__main__':main()
