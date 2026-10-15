import { paperExecute as execute } from "./paperDb.js";

const countCache = new Map();
const COUNT_CACHE_TTL_MS = 30_000;
const pageCache = new Map();
const PAGE_CACHE_TTL_MS = 10_000;
const aggregateCache = new Map();
const AGGREGATE_CACHE_TTL_MS = 30_000;

const PAPER_COLUMNS = `
  id,
  paper_id,
  arxiv_id,
  source,
  conference,
  year,
  subjects,
  primary_subject,
  track,
  title,
  authors,
  abstract,
  pdf_url,
  keywords,
  entry_type,
  editor,
  booktitle,
  pages,
  publisher,
  doi,
  biburl,
  bibsource,
  source_file
`;

function parseJsonArray(value) {
  if (Array.isArray(value)) return value;
  if (!value) return [];
  if (typeof value !== "string") return [];

  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toPaper(row) {
  return {
    id: Number(row.id),
    paper_id: row.paper_id,
    arxiv_id: row.arxiv_id,
    source: row.source,
    conference: row.conference,
    year: Number(row.year),
    subjects: parseJsonArray(row.subjects),
    primary_subject: row.primary_subject,
    track: row.track,
    title: row.title,
    authors: parseJsonArray(row.authors),
    abstract: row.abstract,
    pdf_url: row.pdf_url,
    keywords: parseJsonArray(row.keywords),
    type: row.entry_type,
    editor: parseJsonArray(row.editor),
    booktitle: row.booktitle,
    pages: row.pages,
    publisher: row.publisher,
    doi: row.doi,
    biburl: row.biburl,
    bibsource: row.bibsource,
    source_file: row.source_file
  };
}

function escapeLike(value) {
  return value.replace(/[\\%_]/g, "\\$&");
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

function buildWhere(filters = {}) {
  const clauses = [];
  const params = [];

  if (filters.conference) {
    clauses.push("conference = ?");
    params.push(String(filters.conference).trim());
  }

  if (filters.year) {
    const year = Number(filters.year);
    if (Number.isInteger(year)) {
      clauses.push("year = ?");
      params.push(year);
    }
  }

  if (filters.track) {
    clauses.push("track = ?");
    params.push(String(filters.track).trim());
  }

  if (filters.q) {
    const query = String(filters.q).trim();
    const conferenceQuery = `%${escapeLike(query)}%`;
    clauses.push(`(
      MATCH(title, abstract, authors_text, keywords_text)
        AGAINST (? IN NATURAL LANGUAGE MODE)
      OR conference LIKE ? ESCAPE '\\\\'
    )`);
    params.push(query, conferenceQuery);
  }

  return {
    sql: clauses.length ? `WHERE ${clauses.join(" AND ")}` : "",
    params
  };
}

export async function listPapers(filters = {}) {
  const limit = normalizeLimit(filters.limit);
  const offset = normalizeOffset(filters.offset);
  const where = buildWhere(filters);
  const pageKey = JSON.stringify([where.sql, where.params, limit, offset]);
  const cachedPage = pageCache.get(pageKey);
  if (cachedPage && Date.now() - cachedPage.createdAt < PAGE_CACHE_TTL_MS) {
    return cachedPage.value;
  }

  const cacheKey = JSON.stringify([where.sql, where.params]);
  const cachedCount = countCache.get(cacheKey);
  let total;
  const rowsPromise = execute(
    `SELECT ${PAPER_COLUMNS}
     FROM papers
     ${where.sql}
     ORDER BY year DESC, conference ASC, title ASC, id ASC
     LIMIT ${limit} OFFSET ${offset}`,
    where.params
  );
  if (cachedCount && Date.now() - cachedCount.createdAt < COUNT_CACHE_TTL_MS) {
    total = cachedCount.total;
  } else {
    const countRows = await execute(
      `SELECT COUNT(*) AS total FROM papers ${where.sql}`,
      where.params
    );
    total = Number(countRows[0]?.total || 0);
    countCache.set(cacheKey, { createdAt: Date.now(), total });
    while (countCache.size > 256) countCache.delete(countCache.keys().next().value);
  }
  const rows = await rowsPromise;

  const value = {
    items: rows.map(toPaper),
    total,
    limit,
    offset
  };
  pageCache.set(pageKey, { createdAt: Date.now(), value });
  while (pageCache.size > 512) pageCache.delete(pageCache.keys().next().value);
  return value;
}

export async function listPaperTracks(conference) {
  const key = `tracks:${String(conference).trim()}`;
  const cached = aggregateCache.get(key);
  if (cached && Date.now() - cached.createdAt < AGGREGATE_CACHE_TTL_MS) return cached.value;
  const rows = await execute(
    `SELECT year, COALESCE(NULLIF(track, ''), 'Unknown') AS track, COUNT(*) AS count
     FROM papers
     WHERE conference = ?
     GROUP BY year, track
     ORDER BY year DESC, track ASC`,
    [conference]
  );

  const value = rows.map((row) => ({
    year: Number(row.year),
    track: row.track,
    count: Number(row.count)
  }));
  aggregateCache.set(key, { createdAt: Date.now(), value });
  return value;
}

export async function listPaperVenues() {
  const cached = aggregateCache.get("venues");
  if (cached && Date.now() - cached.createdAt < AGGREGATE_CACHE_TTL_MS) return cached.value;
  const rows = await execute(
    `SELECT
       conference,
       COUNT(*) AS count,
       MIN(year) AS min_year,
       MAX(year) AS max_year,
       COUNT(DISTINCT year) AS year_count,
       COUNT(DISTINCT NULLIF(track, '')) AS track_count,
       GROUP_CONCAT(DISTINCT year ORDER BY year DESC) AS years_csv
     FROM papers
     GROUP BY conference
     ORDER BY conference ASC`
  );

  const value = rows.map((row) => ({
    conference: row.conference,
    count: Number(row.count),
    minYear: Number(row.min_year),
    maxYear: Number(row.max_year),
    yearCount: Number(row.year_count),
    trackCount: Number(row.track_count),
    years: String(row.years_csv || "")
      .split(",")
      .filter(Boolean)
      .map((year) => Number(year))
  }));
  aggregateCache.set("venues", { createdAt: Date.now(), value });
  return value;
}
