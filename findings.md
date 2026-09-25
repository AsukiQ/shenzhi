# Findings

- SearchResult is shared by search flows and serialized directly by HTTP.
- Existing SQLite has source_id and paper_id but no arxiv_id column.
- Disk has 211 MiB free; avoid database rebuild or large temporary files.
- User example exists as paper:2305.14299v3, source_id null; normalize explicit source_id then explicit arXiv paper_id, otherwise null. No graph calls or index rebuild required.
