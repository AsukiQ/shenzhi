import neo4j from "neo4j-driver";
import { fetchEnrichedKg } from "./kgEnrichedRead.js";
import { runKgEnrich } from "./kgEnrich.js";

const COLORS = {
  Paper: ["#3b82f6", "#93c5fd"],
  Author: ["#10b981", "#6ee7b7"],
  Institution: ["#14b8a6", "#99f6e4"],
  Method: ["#f97316", "#fdba74"],
  Funding: ["#eab308", "#fde047"],
  Topic: ["#8b5cf6", "#c4b5fd"],
  Conference: ["#ef4444", "#fca5a5"],
  Venue: ["#ef4444", "#fca5a5"],
};

const REL_COLORS = {
  CITES: "#3b82f6",
  AUTHORED_BY: "#10b981",
  AFFILIATED_WITH: "#14b8a6",
  PROPOSES: "#f97316",
  USES_AS_BASELINE: "#fb923c",
  FUNDED_BY: "#eab308",
  HAS_TOPIC: "#8b5cf6",
  PUBLISHED_IN: "#ef4444",
  PART_OF: "#fca5a5"
};

const NODE_TYPES = ["Paper", "Author", "Institution", "Method", "Funding", "Topic", "Conference", "Venue"];
const NODE_ID_KEYS = {
  Paper: "paper_id",
  Author: "author_id",
  Institution: "institution_id",
  Method: "method_id",
  Funding: "funding_id",
  Topic: "topic_id",
  Conference: "conference_id",
  Venue: "venue_id"
};
const PAPER_DETAIL_RELS = ["AUTHORED_BY", "PROPOSES", "USES_AS_BASELINE", "FUNDED_BY", "HAS_TOPIC", "PUBLISHED_IN"];
const CONTEXT_DETAIL_RELS = ["AFFILIATED_WITH", "PART_OF"];
const MAX_CITATION_NEIGHBORS = 80;
const MAX_CITATION_LINES = 400;
const MAX_DETAIL_LINES = 320;
const MAX_CONTEXT_LINES = 160;
const emptyKgSearchCache = new Map();

const driver = neo4j.driver(
  process.env.NEO4J_URI || "bolt://neo4j:7687",
  neo4j.auth.basic(process.env.NEO4J_USER || "neo4j", process.env.NEO4J_PASSWORD || "")
);

function toNative(value) {
  if (neo4j.isInt(value)) return value.toNumber();
  if (Array.isArray(value)) return value.map(toNative);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, toNative(item)]));
  }
  return value;
}

function nodeType(node) {
  return NODE_TYPES.find((label) => node.labels?.includes(label)) || "Paper";
}

function graphNodeId(type, props) {
  const key = NODE_ID_KEYS[type];
  return (key && props[key]) || `${type}_${props.name || props.title || props.year || "unknown"}`;
}

function graphNodeText(type, props) {
  return props.title || props.name || String(props.year || graphNodeId(type, props));
}

function compactDescription(type, props, text) {
  const description = type === "Paper"
    ? `来源: ${props.venue || ""} ${props.year || ""}`.trim()
    : String(props.content || text || "");
  return description.length > 400 ? `${description.slice(0, 397)}...` : description;
}

function formatNode(node, extra = {}) {
  const type = nodeType(node);
  const props = toNative(node.properties || {});
  const [color, borderColor] = COLORS[type] || COLORS.Paper;
  const id = graphNodeId(type, props);
  const text = graphNodeText(type, props);

  return {
    id,
    text,
    color,
    borderColor,
    data: {
      type,
      description: compactDescription(type, props, text),
      labels: ["__PaperKG__", type],
      abstract: props.abstract || "",
      authors: extra.authors || [],
      name: props.name || "",
      year: props.year ?? null,
      venue: props.venue || "",
      doi: props.doi || "",
      pdf_url: props.pdf_url || "",
      url: props.url || "",
      external: props.external === true
    }
  };
}

function paperSummary(node, authors = []) {
  const props = toNative(node.properties || {});
  return {
    paper_id: props.paper_id,
    title: props.title || props.paper_id,
    authors: toNative(authors).filter(Boolean),
    year: props.year ?? null,
    venue: props.venue || "",
    abstract: props.abstract || "",
    doi: props.doi || "",
    pdf_url: props.pdf_url || ""
  };
}

function formatLine(fromNode, toNode, type) {
  const fromType = nodeType(fromNode);
  const toType = nodeType(toNode);
  const fromProps = toNative(fromNode.properties || {});
  const toProps = toNative(toNode.properties || {});
  const from = graphNodeId(fromType, fromProps);
  const to = graphNodeId(toType, toProps);

  return {
    from,
    to,
    text: type,
    color: REL_COLORS[type] || "#64748b",
    data: {
      type,
      fromText: graphNodeText(fromType, fromProps),
      toText: graphNodeText(toType, toProps),
      description: type,
      weight: 1
    }
  };
}

export function registerKgRoutes(app) {

  app.get("/api/kg/enriched", async (req, res) => {
    try {
      const arxivId = typeof req.query.arxivId === "string" ? req.query.arxivId.trim() : "";
      const result = await fetchEnrichedKg(arxivId);
      if (!result) return res.status(404).json({ message: "Enriched KG not found" });
      res.json(result);
    } catch (error) {
      res.status(error.statusCode || 500).json({
        message: "Failed to load enriched KG",
        error: error instanceof Error ? error.message : String(error)
      });
    }
  });


  app.post("/api/kg/enrich", async (req, res) => {
    try {
      const result = await runKgEnrich({
        arxivHtmlUrl: req.body?.arxivHtmlUrl,
        dryRun: req.body?.dryRun !== false,
        maxSections: req.body?.maxSections
      });
      res.json(result);
    } catch (error) {
      res.status(error.statusCode || 500).json({
        message: "Failed to enrich paper KG",
        error: error instanceof Error ? error.message : String(error)
      });
    }
  });

  app.get("/api/kg/search", async (req, res) => {
    const q = typeof req.query.q === "string" ? req.query.q.trim() : "";
    const limit = Math.min(Math.max(Number(req.query.limit) || 20, 1), 50);
    const session = driver.session();

    try {
      const legacySearch = () => session.executeRead((tx) =>
        tx.run(`
          MATCH (p:Paper)
          WHERE p.external = false
            AND ($q = "" OR toLower(coalesce(p.title, "")) CONTAINS toLower($q))
          WITH p
          ORDER BY p.year DESC, p.title
          LIMIT $limit
          OPTIONAL MATCH (p)-[ar:AUTHORED_BY]->(a:Author)
          WITH p, ar, a ORDER BY coalesce(ar.position, ar.author_order, 2147483647), a.name
          WITH p, [name IN collect(a.name) WHERE name IS NOT NULL] AS authors
          RETURN p, authors
          ORDER BY p.year DESC, p.title
        `, { q, limit: neo4j.int(limit) })
      );
      let result;
      if (!q) {
        const cached = emptyKgSearchCache.get(limit);
        if (cached && Date.now() - cached.createdAt < 30000) {
          return res.json(cached.value);
        }
        result = await legacySearch();
      } else {
        try {
          result = await session.executeRead((tx) =>
            tx.run(`
              CALL db.index.fulltext.queryNodes("paper_title_fulltext", $q)
              YIELD node AS p, score
              WHERE p.external = false
              WITH p, score
              ORDER BY score DESC, coalesce(p.year, 0) DESC, p.title
              LIMIT $limit
              OPTIONAL MATCH (p)-[ar:AUTHORED_BY]->(a:Author)
              WITH p, ar, a ORDER BY coalesce(ar.position, ar.author_order, 2147483647), a.name
              WITH p, [name IN collect(a.name) WHERE name IS NOT NULL] AS authors
              RETURN p, authors
            `, { q, limit: neo4j.int(limit) })
          );
        } catch (error) {
          if (error?.code !== "Neo.ClientError.Schema.IndexNotFound") throw error;
          result = await legacySearch();
        }
      }

      const payload = result.records.map((record) => paperSummary(record.get("p"), record.get("authors")));
      if (!q) {
        emptyKgSearchCache.set(limit, { createdAt: Date.now(), value: payload });
      }
      res.json(payload);
    } catch (error) {
      res.status(500).json({ message: "Failed to search KG papers", error: error instanceof Error ? error.message : String(error) });
    } finally {
      await session.close();
    }
  });

  app.get("/api/kg/paper", async (req, res) => {
    const paperId = typeof req.query.paperId === "string" ? req.query.paperId.trim() : "";
    if (!paperId) return res.status(400).json({ message: "paperId is required" });

    const session = driver.session();
    try {
      const result = await session.executeRead((tx) =>
        tx.run(`
          MATCH (p:Paper {paper_id: $paperId})
          CALL {
            WITH p
            OPTIONAL MATCH (p)-[ar:AUTHORED_BY]->(a:Author)
            WITH a, ar ORDER BY coalesce(ar.position, ar.author_order, 2147483647), a.name
            RETURN [name IN collect(a.name) WHERE name IS NOT NULL] AS authors
          }
          RETURN p, authors
          LIMIT 1
        `, { paperId })
      );

      if (!result.records.length) return res.status(404).json({ message: "Paper not found" });
      res.json(paperSummary(result.records[0].get("p"), result.records[0].get("authors")));
    } catch (error) {
      res.status(500).json({ message: "Failed to load KG paper", error: error instanceof Error ? error.message : String(error) });
    } finally {
      await session.close();
    }
  });

  app.get("/api/kg/graph", async (req, res) => {
    const paperId = typeof req.query.paperId === "string" ? req.query.paperId.trim() : "";
    const depth = Math.min(Math.max(Number(req.query.depth) || 2, 1), 3);
    if (!paperId) return res.status(400).json({ message: "paperId is required" });

    const session = driver.session();
    try {
      const papers = await session.executeRead(async (tx) => {
        const rootResult = await tx.run(`
          MATCH (root:Paper {paper_id: $paperId})
          RETURN root
          LIMIT 1
        `, { paperId });
        if (!rootResult.records.length) return null;

        const selected = [rootResult.records[0].get("root")];
        const seenIds = new Set([paperId]);
        let frontierIds = [paperId];

        for (let level = 0; level < depth && frontierIds.length && selected.length <= MAX_CITATION_NEIGHBORS; level += 1) {
          const remaining = MAX_CITATION_NEIGHBORS + 1 - selected.length;
          const candidateLimit = Math.min(remaining * 4, MAX_CITATION_NEIGHBORS * 4);
          const neighborResult = await tx.run(`
            MATCH (frontier:Paper)-[:CITES]-(neighbor:Paper)
            WHERE frontier.paper_id IN $frontierIds
              AND NOT (neighbor.paper_id IN $seenIds)
            RETURN neighbor
            LIMIT $candidateLimit
          `, {
            frontierIds,
            seenIds: [...seenIds],
            candidateLimit: neo4j.int(candidateLimit)
          });

          const nextFrontier = [];
          for (const record of neighborResult.records) {
            const neighbor = record.get("neighbor");
            const neighborId = toNative(neighbor.properties || {}).paper_id;
            if (!neighborId || seenIds.has(neighborId)) continue;
            seenIds.add(neighborId);
            selected.push(neighbor);
            nextFrontier.push(neighborId);
            if (selected.length > MAX_CITATION_NEIGHBORS) break;
          }
          frontierIds = nextFrontier;
        }

        return selected;
      });

      if (!papers) return res.status(404).json({ message: "Paper not found" });

      const paperIds = papers.map((paper) => toNative(paper.properties).paper_id).filter(Boolean);
      const nodes = new Map();
      const lines = new Map();
      const authorMap = new Map();
      const authorIds = new Set();
      const conferenceIds = new Set();

      for (const paper of papers) {
        const node = formatNode(paper);
        nodes.set(node.id, node);
      }

      const citeResult = await session.executeRead((tx) =>
        tx.run(`
          MATCH (a:Paper)-[r:CITES]->(b:Paper)
          WHERE a.paper_id IN $paperIds AND b.paper_id IN $paperIds
          RETURN a, b
          LIMIT $maxLines
        `, { paperIds, maxLines: neo4j.int(MAX_CITATION_LINES) })
      );

      for (const record of citeResult.records) {
        const line = formatLine(record.get("a"), record.get("b"), "CITES");
        lines.set(`${line.from}|${line.to}|${line.text}`, line);
      }

      const detailResult = await session.executeRead((tx) =>
        tx.run(`
          MATCH (p:Paper)-[r]->(n)
          WHERE p.paper_id IN $paperIds AND type(r) IN $detailRels
          WITH p, r, n, CASE WHEN p.paper_id = $rootPaperId THEN 0 ELSE 1 END AS rootPriority
          ORDER BY rootPriority
          RETURN p, type(r) AS relType, r, n
          LIMIT $maxLines
        `, {
          paperIds,
          rootPaperId: paperId,
          detailRels: PAPER_DETAIL_RELS,
          maxLines: neo4j.int(MAX_DETAIL_LINES)
        })
      );

      for (const record of detailResult.records) {
        const paper = record.get("p");
        const target = record.get("n");
        const relType = record.get("relType");
        const relProps = toNative(record.get("r").properties || {});
        const targetNode = formatNode(target);
        const paperNode = formatNode(paper);

        nodes.set(targetNode.id, targetNode);
        const line = formatLine(paper, target, relType);
        lines.set(`${line.from}|${line.to}|${line.text}`, line);

        if (relType === "AUTHORED_BY") {
          const list = authorMap.get(paperNode.id) || [];
          list.push({
            name: targetNode.text,
            position: Number(relProps.position ?? relProps.author_order ?? Number.MAX_SAFE_INTEGER)
          });
          authorMap.set(paperNode.id, list);
          authorIds.add(targetNode.id);
        } else if (relType === "PUBLISHED_IN") {
          conferenceIds.add(targetNode.id);
        }
      }

      if (authorIds.size || conferenceIds.size) {
        const contextResult = await session.executeRead((tx) =>
          tx.run(`
            MATCH (source)-[r]->(target)
            WHERE type(r) IN $contextRels
              AND ((type(r) = "AFFILIATED_WITH" AND source:Author AND source.author_id IN $authorIds)
                OR (type(r) = "PART_OF" AND source:Conference AND source.conference_id IN $conferenceIds))
            RETURN source, type(r) AS relType, target
            LIMIT $maxLines
          `, {
            authorIds: [...authorIds],
            conferenceIds: [...conferenceIds],
            contextRels: CONTEXT_DETAIL_RELS,
            maxLines: neo4j.int(MAX_CONTEXT_LINES)
          })
        );

        for (const record of contextResult.records) {
          const source = record.get("source");
          const target = record.get("target");
          const relType = record.get("relType");
          const sourceNode = formatNode(source);
          const targetNode = formatNode(target);
          const line = formatLine(source, target, relType);
          nodes.set(sourceNode.id, sourceNode);
          nodes.set(targetNode.id, targetNode);
          lines.set(`${line.from}|${line.to}|${line.text}`, line);
        }
      }

      for (const [id, authorEntries] of authorMap.entries()) {
        const node = nodes.get(id);
        if (node) {
          node.data.authors = authorEntries
            .sort((left, right) => left.position - right.position || left.name.localeCompare(right.name))
            .map((entry) => entry.name);
        }
      }

      res.json({
        rootId: paperId,
        nodes: [...nodes.values()],
        lines: [...lines.values()]
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to load KG graph", error: error instanceof Error ? error.message : String(error) });
    } finally {
      await session.close();
    }
  });
}
