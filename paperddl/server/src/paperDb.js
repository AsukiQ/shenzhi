import "./env.js";
import path from "node:path";
import Database from "better-sqlite3";

// Paper metadata and the FTS5 index are a local, read-mostly SQLite database.
// The path is explicit in production so the Node API and retrieval service use
// the same canonical index.
const defaultPath = "/home/user2/jiansuo/retrieval_backend/papers_fts.db";
export const paperSqlitePath = path.resolve(
  process.env.PAPER_SQLITE_PATH || defaultPath
);

export const paperDb = new Database(paperSqlitePath, {
  readonly: true,
  fileMustExist: true,
  timeout: 5000
});
paperDb.pragma("query_only = ON");

const HEALTH_CACHE_TTL_MS = 15_000;
let healthCache = null;

export function paperPrepare(sql) {
  return paperDb.prepare(sql);
}

export function paperHealthcheck() {
  const now = Date.now();
  if (healthCache && now - healthCache.createdAt < HEALTH_CACHE_TTL_MS) {
    return { ...healthCache.value };
  }

  const counts = paperDb.prepare(
    "SELECT record_type, COUNT(*) AS total FROM papers GROUP BY record_type"
  ).all();
  const recordCounts = Object.fromEntries(
    counts.map((row) => [row.record_type || "canonical", Number(row.total)])
  );
  const paperCount = Object.values(recordCounts).reduce((sum, value) => sum + value, 0);
  const canonicalCount = recordCounts.canonical || 0;
  const value = {
    status: "ok",
    backend: "sqlite_fts5",
    paperCount,
    distinctPaperCount: paperCount,
    canonicalCount,
    recordCounts,
    path: paperSqlitePath
  };
  healthCache = { createdAt: now, value };
  return { ...value, recordCounts: { ...recordCounts } };
}

export function closePaperDb() {
  if (paperDb.open) paperDb.close();
}
