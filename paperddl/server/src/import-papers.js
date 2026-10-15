import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const serverRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(serverRoot, "..");
const defaultJsonPath = path.join(repoRoot, "pre4data", "merged_all_results.json");
const jsonPathArg = process.argv.find((arg) => !arg.startsWith("--") && arg.endsWith(".json"));
const shouldAppend = process.argv.includes("--append");
const jsonPath = path.resolve(process.cwd(), jsonPathArg || defaultJsonPath);
const dbPath = path.resolve(process.env.PAPER_SQLITE_PATH || "/home/user2/jiansuo/retrieval_backend/papers_fts.db");

function asString(value) {
  return value === null || value === undefined ? "" : String(value);
}
function asStringArray(value) {
  return Array.isArray(value) ? value.map(asString).filter(Boolean) : [];
}
function getTrack(subjects) {
  const primary = subjects[0] || "";
  const [, suffix] = primary.split(" - ");
  return suffix?.trim() || primary.trim() || "Unknown";
}
function ftsText(value) {
  return asString(value).toLowerCase().match(/[a-z0-9][a-z0-9_+.-]*/g)?.join(" ") || "";
}
function normalizeRecord(record) {
  const subjects = asStringArray(record.subjects);
  const authors = asStringArray(record.authors);
  const keywords = asStringArray(record.keywords);
  const editor = asStringArray(record.editor);
  return {
    paper_id: asString(record.paper_id),
    source_id: asString(record.source_id || record.paper_id),
    arxiv_id: asString(record.arxiv_id),
    source: asString(record.source),
    conference: asString(record.conference),
    year: Number(record.year) || null,
    subjects_json: JSON.stringify(subjects),
    primary_subject: asString(record.primary_subject || subjects[0]),
    track: asString(record.track || getTrack(subjects)),
    title: asString(record.title),
    authors_json: JSON.stringify(authors),
    abstract: asString(record.abstract),
    pdf_url: asString(record.pdf_url),
    keywords_json: JSON.stringify(keywords),
    entry_type: asString(record.type || record.entry_type),
    editor_json: JSON.stringify(editor),
    booktitle: asString(record.booktitle),
    pages: asString(record.pages),
    publisher: asString(record.publisher),
    doi: asString(record.doi),
    biburl: asString(record.biburl),
    bibsource: asString(record.bibsource),
    source_file: asString(record.source_file),
    funding_json: JSON.stringify(asStringArray(record.funding)),
    external: record.external === true ? 1 : 0,
    record_type: record.external === true ? "external_full" : "canonical"
  };
}

async function main() {
  console.log(`[papers:import] reading ${jsonPath}`);
  const records = JSON.parse(await fs.readFile(jsonPath, "utf8"));
  if (!Array.isArray(records)) throw new Error("Expected a top-level JSON array");
  const db = new Database(dbPath, { timeout: 30_000 });
  try {
    db.pragma("journal_mode = WAL");
    const columns = new Set(db.prepare("PRAGMA table_info(papers)").all().map((row) => row.name));
    const additions = {
      external: "INTEGER NOT NULL DEFAULT 1", record_type: "TEXT NOT NULL DEFAULT 'external_full'",
      arxiv_id: "TEXT NOT NULL DEFAULT ''", source: "TEXT NOT NULL DEFAULT ''",
      primary_subject: "TEXT NOT NULL DEFAULT ''", track: "TEXT NOT NULL DEFAULT ''",
      pdf_url: "TEXT NOT NULL DEFAULT ''", entry_type: "TEXT NOT NULL DEFAULT ''",
      editor_json: "TEXT NOT NULL DEFAULT '[]'", booktitle: "TEXT NOT NULL DEFAULT ''",
      pages: "TEXT NOT NULL DEFAULT ''", publisher: "TEXT NOT NULL DEFAULT ''",
      doi: "TEXT NOT NULL DEFAULT ''", biburl: "TEXT NOT NULL DEFAULT ''",
      bibsource: "TEXT NOT NULL DEFAULT ''", source_file: "TEXT NOT NULL DEFAULT ''",
      image_id: "TEXT"
    };
    for (const [name, definition] of Object.entries(additions)) {
      if (!columns.has(name)) db.exec(`ALTER TABLE papers ADD COLUMN ${name} ${definition}`);
    }
    if (!shouldAppend) {
      db.exec("DELETE FROM paper_authors; DELETE FROM paper_keywords; DELETE FROM paper_subjects; DELETE FROM paper_funding; DELETE FROM papers_fts; DELETE FROM papers;");
    }
    const upsert = db.prepare(`
      INSERT INTO papers (paper_id, source_id, title, abstract, conference, year,
        authors_json, keywords_json, subjects_json, funding_json, external, record_type,
        arxiv_id, source, primary_subject, track, pdf_url, entry_type, editor_json,
        booktitle, pages, publisher, doi, biburl, bibsource, source_file)
      VALUES (@paper_id, @source_id, @title, @abstract, @conference, @year,
        @authors_json, @keywords_json, @subjects_json, @funding_json, @external, @record_type,
        @arxiv_id, @source, @primary_subject, @track, @pdf_url, @entry_type, @editor_json,
        @booktitle, @pages, @publisher, @doi, @biburl, @bibsource, @source_file)
      ON CONFLICT(paper_id) DO UPDATE SET
        source_id=excluded.source_id, title=excluded.title, abstract=excluded.abstract,
        conference=excluded.conference, year=excluded.year, authors_json=excluded.authors_json,
        keywords_json=excluded.keywords_json, subjects_json=excluded.subjects_json,
        funding_json=excluded.funding_json, external=excluded.external, record_type=excluded.record_type,
        arxiv_id=excluded.arxiv_id, source=excluded.source, primary_subject=excluded.primary_subject,
        track=excluded.track, pdf_url=excluded.pdf_url, entry_type=excluded.entry_type,
        editor_json=excluded.editor_json, booktitle=excluded.booktitle, pages=excluded.pages,
        publisher=excluded.publisher, doi=excluded.doi, biburl=excluded.biburl,
        bibsource=excluded.bibsource, source_file=excluded.source_file
    `);
    const replaceFts = db.prepare("DELETE FROM papers_fts WHERE paper_id = ?");
    const insertFts = db.prepare("INSERT INTO papers_fts(paper_id,title,abstract,keywords,subjects) VALUES (?,?,?,?,?)");
    const deleteRelations = db.transaction((paperId) => {
      db.prepare("DELETE FROM paper_authors WHERE paper_id = ?").run(paperId);
      db.prepare("DELETE FROM paper_keywords WHERE paper_id = ?").run(paperId);
      db.prepare("DELETE FROM paper_subjects WHERE paper_id = ?").run(paperId);
      db.prepare("DELETE FROM paper_funding WHERE paper_id = ?").run(paperId);
    });
    const transaction = db.transaction((record) => {
      const row = normalizeRecord(record);
      if (!row.paper_id) return false;
      upsert.run(row);
      replaceFts.run(row.paper_id);
      insertFts.run(
        row.paper_id,
        ftsText(row.title),
        ftsText(row.abstract),
        ftsText(asStringArray(record.keywords).join(" ")),
        ftsText([...asStringArray(record.subjects), ...asStringArray(record.funding)].join(" "))
      );
      deleteRelations(row.paper_id);
      for (const [table, values] of [["paper_authors", asStringArray(record.authors)], ["paper_keywords", asStringArray(record.keywords)], ["paper_subjects", asStringArray(record.subjects)], ["paper_funding", asStringArray(record.funding)]]) {
        const stmt = db.prepare(`INSERT OR IGNORE INTO ${table}(paper_id,value,value_key,position) VALUES (?,?,?,?)`);
        values.forEach((value, position) => stmt.run(row.paper_id, value, value.toLowerCase(), position));
      }
      return true;
    });
    let imported = 0;
    for (const record of records) {
      if (transaction(record)) imported += 1;
      if (imported % 2000 === 0) console.log(`[papers:import] imported ${imported}/${records.length}`);
    }
    db.pragma("wal_checkpoint(TRUNCATE)");
    console.log(JSON.stringify({ db: dbPath, imported, total: records.length }));
  } finally {
    db.close();
  }
}

main().catch((error) => {
  console.error("[papers:import] failed:", error);
  process.exitCode = 1;
});
