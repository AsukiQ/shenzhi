"""Validate and upsert OpenAlex author ZIPs; report exact DOI matches without merging authors."""
import argparse
import hashlib
import json
import re
import sqlite3
import zipfile
from pathlib import Path
from load_product_metadata import load


def doi_key(value):
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', str(value or '').strip(), flags=re.I).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    records = {}
    sources = []
    duplicate_count = 0
    for archive in args.archives:
        count = 0
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not member.filename.endswith('.json') or member.filename.startswith('__MACOS/'):
                    continue
                if member.file_size > 10_000_000:
                    raise ValueError('Oversized author JSON')
                row = json.loads(z.read(member))
                aid = str(row.get('id', '')).rsplit('/', 1)[-1]
                if not re.fullmatch(r'A\d+', aid) or not row.get('display_name'):
                    raise ValueError(f'Invalid author: {member.filename}')
                if aid in records:
                    duplicate_count += 1
                    if records[aid] != row:
                        raise ValueError(f'Conflicting duplicate author: {aid}')
                records[aid] = row
                count += 1
        sources.append({'file': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'records': count})
    profiles = []
    doi_authors = {}
    for aid, row in records.items():
        for work in row.get('works') or []:
            key = doi_key(work.get('doi'))
            if key:
                doi_authors.setdefault(key, set()).add(aid)
        profiles.append({
            'scholar_id': 'openalex:' + aid,
            'name_en': row['display_name'],
            'avatar_path': row.get('avatar_path') or None,
            'affiliations': [{'institution': i['display_name'], 'position': ''}
                             for i in row.get('last_known_institutions') or [] if i.get('display_name')],
            'metrics': {'citation_count': row.get('cited_by_count')},
            'affiliation_history': [
                {'institution_id': a.get('institution', {}).get('id'),
                 'institution': a.get('institution', {}).get('display_name'),
                 'years': sorted(set(a.get('years') or [])), 'position': None,
                 'source': 'OpenAlex', 'basis': 'publication_affiliation'}
                for a in row.get('affiliations') or [] if a.get('institution', {}).get('display_name')],
            'research_impact': {
                'works_count': row.get('works_count'),
                'citation_count': row.get('cited_by_count'),
                'h_index': (row.get('summary_stats') or {}).get('h_index'),
                'i10_index': (row.get('summary_stats') or {}).get('i10_index'),
                'two_year_mean_citedness': (row.get('summary_stats') or {}).get('2yr_mean_citedness'),
                'source': 'OpenAlex', 'coverage': 'OpenAlex author record and works',
                'as_of': None, 'source_updated_at': row.get('updated_date')},
            'annual_citations': [{'year': x['year'], 'count': x['cited_by_count']}
                                 for x in row.get('counts_by_year') or [] if x.get('cited_by_count') is not None],
            'citation_meta': {'source': 'OpenAlex', 'coverage': 'OpenAlex author record and works', 'as_of': None},
            # Source timestamps lack a timezone; preserve them without inventing one.
            'updated_at': row.get('updated_date'),
        })
    matches = []
    with sqlite3.connect(f'file:{args.db}?mode=ro', uri=True) as c:
        before = c.execute('SELECT count(*) FROM scholar_profiles').fetchone()[0]
        for pid, doi, kind in c.execute("SELECT paper_id,doi,record_type FROM papers WHERE doi IS NOT NULL AND doi<>''"):
            key = doi_key(doi)
            if key in doi_authors:
                for aid in sorted(doi_authors[key]):
                    matches.append({'scholar_id': 'openalex:' + aid, 'paper_id': pid, 'doi': key, 'record_type': kind})
    args.output.mkdir(parents=True, exist_ok=True)
    payload = args.output / 'metadata.json'
    payload.write_text(json.dumps({'scholars': profiles}, ensure_ascii=False, indent=2))
    result = load(args.db, payload)
    with sqlite3.connect(args.db) as c:
        for aid in records:
            c.execute("DELETE FROM scholar_paper_links WHERE scholar_id=? AND source='OpenAlex DOI'", ('openalex:' + aid,))
        c.executemany('INSERT OR IGNORE INTO scholar_paper_links(scholar_id,paper_id,source) VALUES (?,?,?)',
                      [(m['scholar_id'], m['paper_id'], 'OpenAlex DOI') for m in matches
                       if m['record_type'] in ('canonical', 'external_full')])
    with sqlite3.connect(f'file:{args.db}?mode=ro', uri=True) as c:
        after = c.execute('SELECT count(*) FROM scholar_profiles').fetchone()[0]
    report = {'sources': sources, 'unique_authors': len(records), 'duplicates': duplicate_count,
              'before': before, 'after': after, 'new_profiles': after-before, 'upserts': result,
              'distinct_input_dois': len(doi_authors), 'doi_matches': matches,
              'note': 'Exact DOI links for canonical/external_full papers stored in shared SQLite; no Neo4j author identities merged. Citation as_of unknown; updated_date is not a statistics cutoff.'}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k,v in report.items() if k != 'doi_matches'}, ensure_ascii=False))
    print('doi_matches', len(matches))


if __name__ == '__main__':
    main()
