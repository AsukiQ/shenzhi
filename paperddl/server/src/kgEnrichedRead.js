import neo4j from "neo4j-driver";

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

function normalizeKgNode(node) {
  const props = toNative(node.properties || {});
  return {
    id: props.node_id,
    type: props.type,
    name: props.name,
    aliases: props.aliases || [],
    summary: props.summary || "",
    evidence: {
      text: props.evidence || ""
    },
    confidence: props.confidence ?? 0,
    research_value_score: props.research_value_score ?? 0
  };
}

export async function fetchEnrichedKg(arxivId) {
  const id = typeof arxivId === "string" ? arxivId.trim() : "";
  if (!id) {
    const error = new Error("arxivId is required");
    error.statusCode = 400;
    throw error;
  }

  const session = driver.session();
  try {
    const result = await session.executeRead((tx) =>
      tx.run(`
        MATCH (p:Paper {arxiv_id: $arxivId})
        OPTIONAL MATCH (p)-[:HAS_KG_NODE]->(n:KGNode)
        WITH p, collect(DISTINCT n) AS nodes
        OPTIONAL MATCH (s:KGNode {paper_id: $arxivId})-[r:KG_RELATION]->(t:KGNode {paper_id: $arxivId})
        RETURN p, nodes, collect({
          source: s.node_id,
          target: t.node_id,
          type: r.type,
          evidence: r.evidence,
          confidence: r.confidence
        }) AS edges
      `, { arxivId: id })
    );

    if (!result.records.length) return null;

    const record = result.records[0];
    const paper = toNative(record.get("p").properties || {});
    const nodes = record.get("nodes")
      .filter(Boolean)
      .map(normalizeKgNode);
    const edges = record.get("edges")
      .map(toNative)
      .filter((edge) => edge.source && edge.target && edge.type)
      .map((edge) => ({
        source: edge.source,
        target: edge.target,
        type: edge.type,
        evidence: {
          text: edge.evidence || ""
        },
        confidence: edge.confidence ?? 0
      }));

    return {
      paper: {
        arxiv_id: paper.arxiv_id || id,
        title: paper.title || "",
        url: paper.url || "",
        image_url: paper.image_url || "",
        image_path: paper.image_path || ""
      },
      kg: {
        nodes,
        edges
      },
      stats: {
        node_count: nodes.length,
        edge_count: edges.length
      }
    };
  } finally {
    await session.close();
  }
}
