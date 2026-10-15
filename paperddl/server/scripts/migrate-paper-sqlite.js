#!/usr/bin/env node
import fs from "node:fs";
import readline from "node:readline";
import path from "node:path";
import Database from "better-sqlite3";

function arg(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

const dbPath = path.resolve(arg("--db", process.env.PAPER_SQLITE_PATH || "/home/user2/jiansuo/retrieval_backend/papers_fts.db"));
const primaryPath = path.resolve(arg("--primary", "/home/user2/shenzhi-dev-migration/primary_papers.jsonl"));
const apply = process.argv.includes("--apply");

if (!fs.existsSync(dbPath)) throw new Error(`SQLite database not found: ${dbPath}`);
if (!fs.existsSync(primaryPath)) throw new Error(`Primary paper JSONL not found: ${primaryPath}`);

const db = new Database(dbPath, { readonly: !apply, fileMustExist: true, timeout: 30_000 });
try {
  const count = db.prepare("SELECT COUNT(*) AS count FROM papers").get().count;
  if (!apply) {
    console.log(JSON.stringify({ mode: "dry-run", db: dbPath, primary: primaryPath, sqliteRows: Number(count) }));
    process.exit(0);
  }

  db.pragma("journal_mode = WAL");
  const columns = new Set(db.prepare("PRAGMA table_info(papers)").all().map((row) => row.name));
  const additions = {
    external: "INTEGER NOT NULL DEFAULT 1",
    record_type: "TEXT NOT NULL DEFAULT 'external_full'",
    arxiv_id: "TEXT NOT NULL DEFAULT ''",
    source: "TEXT NOT NULL DEFAULT ''",
    primary_subject: "TEXT NOT NULL DEFAULT ''",
    track: "TEXT NOT NULL DEFAULT ''",
    pdf_url: "TEXT NOT NULL DEFAULT ''",
    entry_type: "TEXT NOT NULL DEFAULT ''",
    editor_json: "TEXT NOT NULL DEFAULT '[]'",
    booktitle: "TEXT NOT NULL DEFAULT ''",
    pages: "TEXT NOT NULL DEFAULT ''",
    publisher: "TEXT NOT NULL DEFAULT ''",
    doi: "TEXT NOT NULL DEFAULT ''",
    biburl: "TEXT NOT NULL DEFAULT ''",
    bibsource: "TEXT NOT NULL DEFAULT ''",
    source_file: "TEXT NOT NULL DEFAULT ''",
    image_id: "TEXT"
  };
  for (const [name, definition] of Object.entries(additions)) {
    if (!columns.has(name)) db.exec(`ALTER TABLE papers ADD COLUMN ${name} ${definition}`);
  }
  db.exec("CREATE INDEX IF NOT EXISTS idx_papers_record_type ON papers(record_type)");
  db.exec("CREATE INDEX IF NOT EXISTS idx_papers_record_type_year ON papers(record_type, year)");
  db.exec("CREATE INDEX IF NOT EXISTS idx_papers_record_type_conference_year ON papers(record_type, conference, year)");

  // Rows in the search index are all searchable records. Primary metadata is
  // upgraded below; the remaining rows are indexed external records. Reference
  // stubs that only exist in Neo4j are deliberately absent from this database.
  db.prepare("UPDATE papers SET external = 1, record_type = 'external_full'").run();

  const update = db.prepare(`
    UPDATE papers SET
      external = 0,
      record_type = 'canonical',
      arxiv_id = ?, source = ?, conference = ?, year = ?,
      subjects_json = ?, primary_subject = ?, track = ?, title = ?,
      authors_json = ?, abstract = ?, pdf_url = ?, keywords_json = ?,
      entry_type = ?, editor_json = ?, booktitle = ?, pages = ?,
      publisher = ?, doi = ?, biburl = ?, bibsource = ?, source_file = ?,
      funding_json = ?
    WHERE paper_id = ?
  `);
  const transaction = db.transaction((rows) => {
    let updated = 0;
    let missing = 0;
    for (const record of rows) {
      const paperId = String(record.paper_id || "").trim();
      if (!paperId) continue;
      const result = update.run(
        String(record.arxiv_id || ""),
        String(record.source || ""),
        String(record.conference || ""),
        record.year === null || record.year === undefined || record.year === "" ? null : Number(record.year),
        JSON.stringify(Array.isArray(record.subjects) ? record.subjects : []),
        String(record.primary_subject || ""),
        String(record.track || ""),
        String(record.title || ""),
        JSON.stringify(Array.isArray(record.authors) ? record.authors : []),
        String(record.abstract || ""),
        String(record.pdf_url || ""),
        JSON.stringify(Array.isArray(record.keywords) ? record.keywords : []),
        String(record.type || ""),
        JSON.stringify(Array.isArray(record.editor) ? record.editor : []),
        String(record.booktitle || ""),
        String(record.pages || ""),
        String(record.publisher || ""),
        String(record.doi || ""),
        String(record.biburl || ""),
        String(record.bibsource || ""),
        String(record.source_file || ""),
        JSON.stringify(Array.isArray(record.funding) ? record.funding : []),
        paperId
      );
      if (result.changes) updated += 1; else missing += 1;
    }
    return { updated, missing };
  });

  const rows = [];
  const input = readline.createInterface({ input: fs.createReadStream(primaryPath, "utf8"), crlfDelay: Infinity });
  let processed = 0;
  let report = { updated: 0, missing: 0 };
  for await (const line of input) {
    if (!line.trim()) continue;
    rows.push(JSON.parse(line));
    processed += 1;
    if (rows.length >= 1000) {
      const current = transaction(rows);
      report.updated += current.updated;
      report.missing += current.missing;
      rows.length = 0;
    }
  }
  if (rows.length) {
    const current = transaction(rows);
    report.updated += current.updated;
    report.missing += current.missing;
  }
  db.pragma("wal_checkpoint(TRUNCATE)");
  const counts = db.prepare("SELECT record_type, COUNT(*) AS count FROM papers GROUP BY record_type ORDER BY record_type").all();
  console.log(JSON.stringify({ mode: "apply", db: dbPath, primary: primaryPath, processed, sqliteRows: Number(count), ...report, counts }));
} finally {
  db.close();
}
