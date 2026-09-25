"""Create an auditable paper scope; never mutate the production index."""
import argparse
import json
import sqlite3
from pathlib import Path

VENUES = {
    'aaai',
    'aaai conference on artificial intelligence',
    'fortieth aaai conference on artificial intelligence',
    '40th aaai conference on artificial intelligence',
    'proceedings of the aaai conference on artificial intelligence',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    conn = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    rows = conn.execute('SELECT paper_id, conference FROM papers WHERE year=2026')
    selected = [(pid, venue) for pid, venue in rows
                if ' '.join((venue or '').casefold().split()) in VENUES]
    conn.close()
    counts = {}
    for _, venue in selected:
        counts[venue] = counts.get(venue, 0) + 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'year': 2026, 'conference': 'AAAI', 'paper_count': len(selected),
               'source_venue_counts': counts, 'paper_ids': sorted(pid for pid, _ in selected)}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'paper_ids'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
