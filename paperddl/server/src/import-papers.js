import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execute, pool } from "./db.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const serverRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(serverRoot, "..");

const defaultJsonPath = path.join(repoRoot, "pre4data", "merged_all_results.json");
const jsonPathArg = process.argv.find((arg) => !arg.startsWith("--") && arg.endsWith(".json"));
const shouldAppend = process.argv.includes("--append");
const jsonPath = path.resolve(process.cwd(), jsonPathArg || defaultJsonPath);

const columns = [
  "paper_id",
  "arxiv_id",
  "source",
  "conference",
  "year",
  "subjects",
  "primary_subject",
  "track",
  "title",
  "authors",
  "authors_text",
  "abstract",
  "pdf_url",
  "keywords",
  "keywords_text",
  "entry_type",
  "editor",
  "editor_text",
  "booktitle",
  "pages",
  "publisher",
  "doi",
  "biburl",
  "bibsource",
  "source_file"
];

function asString(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function asStringArray(value) {
  if (!Array.isArray(value)) return [];
  return value.map(asString).filter((item) => item.length > 0);
}

function getTrack(subjects) {
  const primary = subjects[0] || "";
  const [, suffix] = primary.split(" - ");
  return suffix?.trim() || primary.trim() || "Unknown";
}

function normalizeRecord(record) {
  const subjects = asStringArray(record.subjects);
  const authors = asStringArray(record.authors);
  const keywords = asStringArray(record.keywords);
  const editor = asStringArray(record.editor);
  const primarySubject = subjects[0] || "";

  return [
    asString(record.paper_id),
    asString(record.arxiv_id),
    asString(record.source),
    asString(record.conference),
    Number(record.year) || 0,
    JSON.stringify(subjects),
    primarySubject,
    getTrack(subjects),
    asString(record.title),
    JSON.stringify(authors),
    authors.join("; "),
    asString(record.abstract),
    asString(record.pdf_url),
    JSON.stringify(keywords),
    keywords.join("; "),
    asString(record.type),
    JSON.stringify(editor),
    editor.join("; "),
    asString(record.booktitle),
    asString(record.pages),
    asString(record.publisher),
    asString(record.doi),
    asString(record.biburl),
    asString(record.bibsource),
    asString(record.source_file)
  ];
}

async function insertChunk(records) {
  const placeholders = records
    .map(() => `(${columns.map(() => "?").join(", ")})`)
    .join(", ");
  const values = records.flatMap(normalizeRecord);

  await execute(
    `INSERT INTO papers (${columns.join(", ")}) VALUES ${placeholders}`,
    values
  );
}

async function main() {
  console.log(`[papers:import] reading ${jsonPath}`);
  const raw = await fs.readFile(jsonPath, "utf8");
  const records = JSON.parse(raw);

  if (!Array.isArray(records)) {
    throw new Error("Expected the paper JSON file to contain a top-level array.");
  }

  if (!shouldAppend) {
    console.log("[papers:import] truncating papers table");
    await execute("TRUNCATE TABLE papers");
  }

  const chunkSize = 200;
  for (let index = 0; index < records.length; index += chunkSize) {
    const chunk = records.slice(index, index + chunkSize);
    await insertChunk(chunk);
    const imported = Math.min(index + chunk.length, records.length);
    if (imported % 2000 === 0 || imported === records.length) {
      console.log(`[papers:import] imported ${imported}/${records.length}`);
    }
  }

  console.log(`[papers:import] done, imported ${records.length} records`);
}

main()
  .catch((error) => {
    console.error("[papers:import] failed:", error);
    process.exitCode = 1;
  })
  .finally(async () => {
    await pool.end();
  });
