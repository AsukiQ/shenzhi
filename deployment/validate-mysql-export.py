"""Validate the row shape and JSON columns of export-mysql.mjs output."""
import collections
import json
from pathlib import Path
import re
import sys


def values(text):
    result, start, quoted, escaped = [], 0, False, False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
        elif quoted and char == '\\':
            escaped = True
        elif char == "'":
            quoted = not quoted
        elif char == ',' and not quoted:
            result.append(text[start:i].strip())
            start = i + 1
    if quoted or escaped:
        raise ValueError('Unterminated SQL string')
    return result + [text[start:].strip()]


def unquote(text):
    if not (text.startswith("'") and text.endswith("'")):
        raise ValueError('Expected quoted JSON')
    escapes = {'0':'\0', 'n':'\n', 'r':'\r', 'b':'\b', 't':'\t', 'Z':'\x1a'}
    output, escaped = [], False
    for char in text[1:-1]:
        if escaped:
            output.append(escapes.get(char, char))
            escaped = False
        elif char == '\\':
            escaped = True
        else:
            output.append(char)
    return ''.join(output)


def validate(path):
    counts = collections.Counter()
    json_columns = {'subjects', 'authors', 'keywords', 'editor'}
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.startswith('INSERT INTO '):
                continue
            match = re.fullmatch(r'INSERT INTO `([^`]+)` \(([^)]+)\) VALUES \((.*)\);\n?', line)
            if not match:
                raise ValueError(f'Invalid INSERT syntax on line {line_no}')
            table, columns, raw = match.groups()
            columns = [c.strip(' `') for c in columns.split(',')]
            row = values(raw)
            if len(row) != len(columns):
                raise ValueError(f'Wrong column count at line {line_no}: {len(row)} vs {len(columns)}')
            for column, value in zip(columns, row):
                if table == 'papers' and column in json_columns and value != 'NULL':
                    json.loads(unquote(value))
            counts[table] += 1
    manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
    if dict(counts) != manifest['counts']:
        raise ValueError(f'Counts differ from manifest: {dict(counts)}')
    print(json.dumps({'status':'ok', 'counts':dict(counts), 'json_columns_validated':sorted(json_columns)}))


if __name__ == '__main__':
    validate(sys.argv[1])
