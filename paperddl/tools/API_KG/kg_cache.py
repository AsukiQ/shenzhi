import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import (
    EXTRACTION_CACHE_FILE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class LocalExtractionCache:
    def __init__(self, path: Path = EXTRACTION_CACHE_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Local extraction cache is invalid JSON; starting empty.")
            return {}

    def _key(self, paper_id: str, section_id: str, hash_value: str,
             schema_version: str, prompt_version: str, model: str) -> str:
        return "|".join([paper_id, section_id, hash_value, schema_version, prompt_version, model])

    def get_extraction(self, paper_id: str, section_id: str, hash_value: str,
                       schema_version: str, prompt_version: str, model: str) -> Optional[Dict]:
        return self.data.get(self._key(paper_id, section_id, hash_value, schema_version, prompt_version, model))

    def save_extraction(self, paper_id: str, section_id: str, hash_value: str,
                        schema_version: str, prompt_version: str, model: str,
                        kg: Dict) -> None:
        self.data[self._key(paper_id, section_id, hash_value, schema_version, prompt_version, model)] = {
            "paper_id": paper_id,
            "section_id": section_id,
            "content_hash": hash_value,
            "schema_version": schema_version,
            "prompt_version": prompt_version,
            "model": model,
            "status": "success",
            "kg": kg,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        return None


class Neo4jExtractionCache:
    def __init__(self):
        if not NEO4J_URI or not NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD are required for Neo4j cache.")
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.database = NEO4J_DATABASE

    def get_extraction(self, paper_id: str, section_id: str, hash_value: str,
                       schema_version: str, prompt_version: str, model: str) -> Optional[Dict]:
        query = """
        MATCH (:Paper {arxiv_id: $paper_id})-[:HAS_SECTION]->(:Section {section_id: $section_id})
              -[:HAS_EXTRACTION]->(r:ExtractionRun {
                content_hash: $content_hash,
                schema_version: $schema_version,
                prompt_version: $prompt_version,
                model: $model,
                status: "success"
              })
        RETURN r.kg_json AS kg_json
        LIMIT 1
        """
        with self.driver.session(database=self.database) as session:
            record = session.run(query, paper_id=paper_id, section_id=section_id,
                                 content_hash=hash_value, schema_version=schema_version,
                                 prompt_version=prompt_version, model=model).single()
        if not record:
            return None
        return json.loads(record["kg_json"])

    def save_extraction(self, paper_id: str, section_id: str, hash_value: str,
                        schema_version: str, prompt_version: str, model: str,
                        kg: Dict) -> None:
        query = """
        MERGE (p:Paper {arxiv_id: $paper_id})
        MERGE (p)-[:HAS_SECTION]->(s:Section {section_id: $section_id})
        MERGE (s)-[:HAS_EXTRACTION]->(r:ExtractionRun {
          content_hash: $content_hash,
          schema_version: $schema_version,
          prompt_version: $prompt_version,
          model: $model
        })
        SET r.status = "success",
            r.kg_json = $kg_json,
            r.updated_at = datetime()
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, paper_id=paper_id, section_id=section_id,
                        content_hash=hash_value, schema_version=schema_version,
                        prompt_version=prompt_version, model=model,
                        kg_json=json.dumps(kg, ensure_ascii=False))

    def save_paper_kg(self, paper: Dict, kg: Dict) -> None:
        query = """
        MERGE (p:Paper {arxiv_id: $paper_id})
        SET p.title = $title,
            p.url = $url,
            p.image_url = $image_url,
            p.image_path = $image_path,
            p.updated_at = datetime()
        WITH p
        UNWIND $nodes AS node
        MERGE (n:KGNode {paper_id: $paper_id, node_id: node.id})
        SET n.type = node.type,
            n.name = node.name,
            n.aliases = node.aliases,
            n.summary = node.summary,
            n.evidence = node.evidence.text,
            n.confidence = node.confidence,
            n.research_value_score = node.research_value_score
        MERGE (p)-[:HAS_KG_NODE]->(n)
        """
        edge_query = """
        MATCH (p:Paper {arxiv_id: $paper_id})
        UNWIND $edges AS edge
        MATCH (s:KGNode {paper_id: $paper_id, node_id: edge.source})
        MATCH (t:KGNode {paper_id: $paper_id, node_id: edge.target})
        MERGE (s)-[r:KG_RELATION {type: edge.type, paper_id: $paper_id}]->(t)
        SET r.evidence = edge.evidence.text,
            r.confidence = edge.confidence
        """
        with self.driver.session(database=self.database) as session:
            images = paper.get("images", [])
            first_image = images[0] if images else {}
            session.run(query, paper_id=paper["arxiv_id"], title=paper.get("title", ""),
                        url=paper.get("url", ""),
                        image_url=first_image.get("url", ""),
                        image_path=first_image.get("local_path", ""),
                        nodes=kg.get("nodes", []))
            session.run(edge_query, paper_id=paper["arxiv_id"], edges=kg.get("edges", []))

    def close(self) -> None:
        self.driver.close()


def build_cache():
    try:
        cache = Neo4jExtractionCache()
        logger.info("Using Neo4j extraction cache.")
        return cache
    except Exception as exc:
        logger.warning("Using local extraction cache: %s", exc)
        return LocalExtractionCache()
