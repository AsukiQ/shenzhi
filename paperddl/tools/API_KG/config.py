import os
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EXTRACTION_CACHE_FILE = DATA_DIR / "extraction_cache.json"

for dir_path in [DATA_DIR]:
    dir_path.mkdir(exist_ok=True)


# arXiv HTML KG extraction
ARXIV_HTML_TEST_URL = os.getenv("ARXIV_HTML_TEST_URL", "https://arxiv.org/html/2602.22727v1")
ARXIV_HTML_CACHE_FIRST = os.getenv("ARXIV_HTML_CACHE_FIRST", "0") == "1"
KG_SCHEMA_VERSION = os.getenv("KG_SCHEMA_VERSION", "academic_claim_evidence_kg_v2")
KG_PROMPT_VERSION = os.getenv("KG_PROMPT_VERSION", "section_aware_claim_evidence_pe_v4")
KG_MAX_SECTION_CHARS = int(os.getenv("KG_MAX_SECTION_CHARS", "12000"))


# LLM API. Compatible with OpenAI-style chat completion APIs.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-5.1")


# Optional Neo4j cache and sink. If unset, local JSON cache is used.
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
