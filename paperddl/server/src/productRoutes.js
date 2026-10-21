import neo4j from "neo4j-driver";
import { driver } from "./kgRoutes.js";
import {
  getScholarProfile,
  getTopic,
  listScholarPapers,
  searchScholarProfiles,
  scholarLinkedPaperIds,
  listTopicPapers,
  listTopics
} from "./productRepository.js";

function pageParam(value, fallback) {
  const parsed = Number(value);
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    const error = new Error("Invalid pagination"); error.status = 400; throw error;
  }
  return parsed;
}

function productScholar(profile, graph) {
  const row = profile.profile || {};
  const citationCount = row.citation_count === null || row.citation_count === undefined
    ? null : Number(row.citation_count);
  return {
    scholar_id: graph.scholar_id,
    name_zh: row.name_zh ?? null,
    name_en: row.name_en || graph.name || graph.scholar_id,
    avatar_path: row.avatar_path ?? null,
    affiliations: profile.affiliations.map((item) => ({
      institution: item.institution,
      position: item.position || ""
    })),
    bio_zh: row.bio_zh ?? null,
    bio_en: row.bio_en ?? null,
    metrics: {
      paper_count: graph.paper_count,
      citation_count: citationCount
    },
    annual_citations: profile.annualCitations,
    affiliation_history: profile.affiliationHistory,
    research_impact: {
      works_count: null, citation_count: null, h_index: null, i10_index: null,
      two_year_mean_citedness: null, source: null, coverage: null, as_of: null,
      source_updated_at: null, ...profile.researchImpact
    },
    updated_at: row.updated_at ?? null,
    citation_meta: {
      source: row.citation_source ?? null,
      coverage: row.citation_coverage ?? null,
      as_of: row.citation_as_of ?? null
    }
  };
}

async function scholarGraph(scholarId) {
  const session = driver.session();
  try {
    const result = await session.executeRead((tx) => tx.run(`
      MATCH (a:Author {author_id: $scholarId})
      OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
      RETURN a.author_id AS scholar_id,
             coalesce(a.name, a.author_id) AS name,
             count(DISTINCT p) AS paper_count
    `, { scholarId }));
    return result.records.length ? result.records[0].toObject() : null;
  } finally {
    await session.close();
  }
}

async function scholarPaperIds(scholarId) {
  const session = driver.session();
  try {
    const result = await session.executeRead((tx) => tx.run(`
      MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author {author_id: $scholarId})
      WHERE coalesce(p.record_type, CASE WHEN coalesce(p.external, false)
        THEN 'external_full' ELSE 'canonical' END) <> 'reference_stub'
      RETURN DISTINCT p.paper_id AS paper_id, coalesce(p.year, 0) AS year
      ORDER BY year DESC, paper_id
    `, { scholarId }));
    return result.records.map((record) => record.toObject().paper_id).filter(Boolean);
  } finally {
    await session.close();
  }
}

async function searchScholars(query, limit, offset) {
  const session = driver.session();
  try {
    const params = { query: String(query || '').trim(), offset: neo4j.int(offset), limit: neo4j.int(limit) };
    let result;
    try {
      result = await session.executeRead((tx) => tx.run(`
        CALL db.index.fulltext.queryNodes('author_name_fulltext', $query) YIELD node AS a, score
        OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
        WHERE coalesce(p.record_type, CASE WHEN coalesce(p.external, false) THEN 'external_full' ELSE 'canonical' END) <> 'reference_stub'
        RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
               count(DISTINCT p) AS paper_count
        ORDER BY paper_count DESC, name, scholar_id
        SKIP $offset LIMIT $limit
      `, params));
    } catch (error) {
      // Older Neo4j instances may not have the full text index. Keep the
      // product endpoint usable with a case insensitive substring fallback.
      result = await session.executeRead((tx) => tx.run(`
        MATCH (a:Author)
        WHERE toLower(coalesce(a.name, '')) CONTAINS toLower($query)
        OPTIONAL MATCH (p:Paper)-[:AUTHORED_BY]->(a)
        WHERE coalesce(p.record_type, CASE WHEN coalesce(p.external, false) THEN 'external_full' ELSE 'canonical' END) <> 'reference_stub'
        RETURN a.author_id AS scholar_id, coalesce(a.name, a.author_id) AS name,
               count(DISTINCT p) AS paper_count
        ORDER BY paper_count DESC, name, scholar_id
        SKIP $offset LIMIT $limit
      `, params));
    }
    const items = result.records.map((record) => {
      const row = record.toObject();
      return {
        scholar_id: row.scholar_id,
        name_zh: getScholarProfile(row.scholar_id).profile?.name_zh ?? null,
        name_en: getScholarProfile(row.scholar_id).profile?.name_en || row.name,
        avatar_path: getScholarProfile(row.scholar_id).profile?.avatar_path ?? null,
        metrics: { paper_count: Number(row.paper_count?.toNumber?.() ?? row.paper_count ?? 0), citation_count: null }
      };
    });
    const profileItems = searchScholarProfiles(query, {limit, offset}).map((row) => ({
      scholar_id: row.scholar_id,
      name_zh: row.name_zh ?? null,
      name_en: row.name_en || row.scholar_id,
      avatar_path: row.avatar_path ?? null,
      metrics: { paper_count: listScholarPapers(scholarLinkedPaperIds(row.scholar_id), {limit:1}).total, citation_count: row.citation_count == null ? null : Number(row.citation_count) }
    }));
    const seen = new Set(items.map((item) => item.scholar_id));
    return [...profileItems.filter((item) => !seen.has(item.scholar_id)), ...items].slice(0, limit);
  } finally {
    await session.close();
  }
}

export function registerProductRoutes(app) {
  app.get("/api/v1/topics", (req, res) => {
    try {
      const enabledParam = req.query.enabled;
      if (enabledParam !== undefined && !["true","false","all"].includes(enabledParam)) return res.status(400).json({message:"enabled must be true, false or all"});
      const enabled = enabledParam === "all" ? "all" : enabledParam === "false" ? false : true;
      res.json(listTopics({
        enabled,
        category: typeof req.query.category === "string" ? req.query.category.trim() : "",
        limit: pageParam(req.query.limit, 100),
        offset: pageParam(req.query.offset, 0)
      }));
    } catch (error) {
      res.status(error.status || 500).json({ message: "Failed to load product topics", error: error.message });
    }
  });

  app.get("/api/v1/topics/:topicId", (req, res) => {
    try {
      const topic = getTopic(req.params.topicId);
      if (!topic) return res.status(404).json({ message: "Topic not found" });
      return res.json(topic);
    } catch (error) {
      return res.status(error.status || 500).json({ message: "Failed to load product topic", error: error.message });
    }
  });

  app.get("/api/v1/topics/:topicId/papers", (req, res) => {
    try {
      const topic = getTopic(req.params.topicId);
      if (!topic) return res.status(404).json({ message: "Topic not found" });
      return res.json(listTopicPapers(req.params.topicId, {
        limit: pageParam(req.query.limit, 20),
        offset: pageParam(req.query.offset, 0)
      }));
    } catch (error) {
      return res.status(error.status || 500).json({ message: "Failed to load topic papers", error: error.message });
    }
  });

  app.get("/api/v1/scholars/search", async (req, res) => {
    try {
      const limit = Math.min(Math.max(pageParam(req.query.limit, 20), 1), 100);
      const offset = Math.max(pageParam(req.query.offset, 0), 0);
      if (typeof req.query.q !== "string" || !req.query.q.trim()) return res.status(400).json({message:"q is required"});
      const items = await searchScholars(req.query.q, limit, offset);
      return res.json({ items, limit, offset });
    } catch (error) {
      return res.status(error.status || 500).json({ message: "Failed to search product scholars", error: error.message });
    }
  });

  app.get("/api/v1/scholars/:scholarId", async (req, res) => {
    try {
      const scholarId = String(req.params.scholarId || "").trim();
      if (!scholarId) return res.status(400).json({ message: "scholar_id is required" });
      const graph = await scholarGraph(scholarId);
      if (!graph && !getScholarProfile(scholarId).profile) return res.status(404).json({ message: "Scholar not found" });
      const profile = getScholarProfile(scholarId);
      const base = graph || { scholar_id: scholarId, name: profile.profile?.name_en || scholarId, paper_count: 0 };
      base.paper_count = listScholarPapers([...scholarLinkedPaperIds(scholarId), ...(graph ? await scholarPaperIds(scholarId) : [])], {limit:1}).total;
      return res.json(productScholar(profile, base));
    } catch (error) {
      return res.status(error.status || 500).json({ message: "Failed to load product scholar", error: error.message });
    }
  });

  app.get("/api/v1/scholars/:scholarId/papers", async (req, res) => {
    try {
      const scholarId = String(req.params.scholarId || "").trim();
      if (!scholarId) return res.status(400).json({ message: "scholar_id is required" });
      const graph = await scholarGraph(scholarId);
      if (!graph && !getScholarProfile(scholarId).profile) return res.status(404).json({ message: "Scholar not found" });
      return res.json(listScholarPapers([...scholarLinkedPaperIds(scholarId), ...(graph ? await scholarPaperIds(scholarId) : [])], {
        limit: pageParam(req.query.limit, 20),
        offset: pageParam(req.query.offset, 0)
      }));
    } catch (error) {
      return res.status(error.status || 500).json({ message: "Failed to load scholar papers", error: error.message });
    }
  });
}

