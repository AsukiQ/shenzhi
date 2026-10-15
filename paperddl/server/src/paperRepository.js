import { paperPrepare } from "./paperDb.js";

const countCache = new Map();
const COUNT_CACHE_TTL_MS = 30_000;
const pageCache = new Map();
const PAGE_CACHE_TTL_MS = 10_000;
const aggregateCache = new Map();
const AGGREGATE_CACHE_TTL_MS = 30_000;
const RECORD_TYPES = new Set(["canonical", "external_full", "all"]);

function parseJsonArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined || value === "") return [];
  try {
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toPaper(row) {
  return {
    id: Number(row.row_id),
    paper_id: row.paper_id,
    arxiv_id: row.arxiv_id || "",
    source: row.source || row.source_id || "",
    conference: row.conference || "",
    year: row.year === null || row.year === undefined ? null : Number(row.year),
    subjects: parseJsonArray(row.subjects_json),
    primary_subject: row.primary_subject || "",
    track: row.track || "",
    title: row.title || "",
    authors: parseJsonArray(row.authors_json),
    abstract: row.abstract || "",
    pdf_url: row.pdf_url || "",
    keywords: parseJsonArray(row.keywords_json),
    type: row.entry_type || "",
    editor: parseJsonArray(row.editor_json),
    booktitle: row.booktitle || "",
    pages: row.pages || "",
    publisher: row.publisher || "",
    doi: row.doi || "",
    biburl: row.biburl || "",
    bibsource: row.bibsource || "",
    source_file: row.source_file || "",
    funding: parseJsonArray(row.funding_json),
    image_id: row.image_id || null,
    record_type: row.record_type || "external_full",
    external: Number(row.external || 0) === 1
  };
}

function normalizeLimit(value) {
  const limit = Number(value || 100);
  if (!Number.isFinite(limit) || limit < 1) return 100;
  return Math.min(Math.floor(limit), 2000);
}

function normalizeOffset(value) {
  const offset = Number(value || 0);
  if (!Number.isFinite(offset) || offset < 0) return 0;
  return Math.floor(offset);
}

function normalizeRecordType(value) {
  const type = String(value || "canonical").trim().toLowerCase();
  return RECORD_TYPES.has(type) ? type : "canonical";
}

function ftsQuery(value) {
  const tokens = String(value || "").toLowerCase().match(/[a-z0-9][a-z0-9_+.-]*/g) || [];
  const unique = [...new Set(tokens)];
  if (!unique.length) return "";
  return unique.map((token) => `"${token.replaceAll('"', '""')}"`).join(" OR ");
}

function buildWhere(filters = {}) {
  const clauses = [];
  const params = [];
  const recordType = normalizeRecordType(filters.recordType);

  if (recordType === "all") {
    clauses.push("p.record_type IN ('canonical', 'external_full')");
  } else {
    clauses.push("p.record_type = ?");
    params.push(recordType);
  }

  if (filters.conference) {
    clauses.push("p.conference = ?");
    params.push(String(filters.conference).trim());
  }
  if (filters.year) {
    const year = Number(filters.year);
    if (Number.isInteger(year)) {
      clauses.push("p.year = ?");
      params.push(year);
    }
  }
  if (filters.track) {
    clauses.push("p.track = ?");
    params.push(String(filters.track).trim());
  }

  const match = ftsQuery(filters.q);
  if (match) {
    clauses.push("p.paper_id IN (SELECT paper_id FROM papers_fts WHERE papers_fts MATCH ?)");
    params.push(match);
  } else if (String(filters.q || "").trim()) {
    // unicode61 has no useful tokenization for a pure Chinese query. Do not
    // silently turn that request into an unfiltered paper listing.
    clauses.push("1 = 0");
  }

  return { sql: `WHERE ${clauses.join(" AND ")}`, params, recordType };
}

function cachePut(cache, key, value, maxSize) {
  cache.set(key, { createdAt: Date.now(), value });
  while (cache.size > maxSize) cache.delete(cache.keys().next().value);
}

export function listPapers(filters = {}) {
  const limit = normalizeLimit(filters.limit);
  const offset = normalizeOffset(filters.offset);
  const where = buildWhere(filters);
  const pageKey = JSON.stringify([where.sql, where.params, limit, offset]);
  const cachedPage = pageCache.get(pageKey);
  if (cachedPage && Date.now() - cachedPage.createdAt < PAGE_CACHE_TTL_MS) return cachedPage.value;

  const cacheKey = JSON.stringify([where.sql, where.params]);
  const cachedCount = countCache.get(cacheKey);
  const now = Date.now();
  const countRow = cachedCount && now - cachedCount.createdAt < COUNT_CACHE_TTL_MS
    ? { total: cachedCount.value }
    : paperPrepare(`SELECT COUNT(*) AS total FROM papers p ${where.sql}`).get(...where.params);
  const total = Number(countRow.total || 0);
  if (!cachedCount || now - cachedCount.createdAt >= COUNT_CACHE_TTL_MS) cachePut(countCache, cacheKey, total, 256);

  const rows = paperPrepare(`
    SELECT p.* FROM papers p
    ${where.sql}
    -- MySQL's unicode collation places leading typographic punctuation before
    -- ASCII digits. Keep the old paper-page ordering stable after the move.
    ORDER BY p.year DESC, p.conference ASC,
      CASE
        WHEN substr(p.title, 1, 1) IN ('‘', '“') THEN 0
        WHEN substr(p.title, 1, 1) GLOB '[A-Za-z0-9]' THEN 1
        ELSE 2
      END ASC,
      p.title ASC, p.row_id ASC
    LIMIT ? OFFSET ?
  `).all(...where.params, limit, offset);
  const value = { items: rows.map(toPaper), total, limit, offset };
  cachePut(pageCache, pageKey, value, 512);
  return value;
}

export function listPaperTracks(conference, recordType = "canonical") {
  const type = normalizeRecordType(recordType);
  const key = `tracks:${type}:${String(conference).trim()}`;
  const cached = aggregateCache.get(key);
  if (cached && Date.now() - cached.createdAt < AGGREGATE_CACHE_TTL_MS) return cached.value;
  const typeClause = type === "all" ? "record_type IN ('canonical', 'external_full')" : "record_type = ?";
  const params = type === "all" ? [conference] : [type, conference];
  const rows = paperPrepare(`
    SELECT year, COALESCE(NULLIF(track, ''), 'Unknown') AS track, COUNT(*) AS count
    FROM papers WHERE ${typeClause} AND conference = ?
    GROUP BY year, track ORDER BY year DESC, track ASC
  `).all(...params);
  const value = rows.map((row) => ({ year: Number(row.year), track: row.track, count: Number(row.count) }));
  cachePut(aggregateCache, key, value, 256);
  return value;
}

export function listPaperVenues(recordType = "canonical") {
  const type = normalizeRecordType(recordType);
  const key = `venues:${type}`;
  const cached = aggregateCache.get(key);
  if (cached && Date.now() - cached.createdAt < AGGREGATE_CACHE_TTL_MS) return cached.value;
  const typeClause = type === "all" ? "record_type IN ('canonical', 'external_full')" : "record_type = ?";
  const typeParams = type === "all" ? [] : [type];
  const rows = paperPrepare(`
    SELECT conference, COUNT(*) AS count, MIN(year) AS min_year,
           MAX(year) AS max_year, COUNT(DISTINCT year) AS year_count,
           COUNT(DISTINCT NULLIF(track, '')) AS track_count
    FROM papers
    WHERE ${typeClause} AND conference IS NOT NULL AND conference <> ''
    GROUP BY conference ORDER BY conference ASC
  `).all(...typeParams);
  const yearsStatement = paperPrepare(`
    SELECT DISTINCT year FROM papers
    WHERE ${typeClause} AND conference = ? AND year IS NOT NULL
    ORDER BY year DESC
  `);
  const value = rows.map((row) => ({
    conference: row.conference,
    count: Number(row.count),
    minYear: Number(row.min_year),
    maxYear: Number(row.max_year),
    yearCount: Number(row.year_count),
    trackCount: Number(row.track_count),
    years: yearsStatement.all(...typeParams, row.conference).map((item) => Number(item.year))
  }));
  cachePut(aggregateCache, key, value, 32);
  return value;
}
