"""Build paper-image metadata without extracting the image archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import zipfile


def build_manifest(archive, db_path):
    with sqlite3.connect(f'{Path(db_path).resolve().as_uri()}?mode=ro', uri=True) as conn:
        papers = {r[0] for r in conn.execute('SELECT paper_id FROM papers')}
    mapping = {}
    unmatched = []
    seen = set()
    basenames = set()
    count = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                raise ValueError(f'Unsafe archive path: {info.filename}')
            if info.filename in seen:
                raise ValueError(f'Duplicate archive path: {info.filename}')
            seen.add(info.filename)
            if path.name in basenames:
                raise ValueError(f'Duplicate image filename: {path.name}')
            basenames.add(path.name)
            count += 1
            paper_id = 'paper:' + path.stem
            if not re.fullmatch(r'[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?', path.stem) or paper_id not in papers:
                unmatched.append(info.filename)
                continue
            if paper_id in mapping:
                raise ValueError(f'Multiple images for paper: {paper_id}')
            mapping.setdefault(paper_id, []).append({
                'image_id': hashlib.sha256(info.filename.encode('utf-8')).hexdigest(),
                'image_path': info.filename,
                'caption': None,
            })
    return mapping, {'image_count': count, 'matched_papers': len(mapping),
                     'matched_images': sum(map(len, mapping.values())),
                     'unmatched': unmatched}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--db', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    mapping, report = build_manifest(args.archive, args.db)
    target = Path(args.output)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(mapping, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)
    target.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v if k != 'unmatched' else len(v) for k, v in report.items()}))
