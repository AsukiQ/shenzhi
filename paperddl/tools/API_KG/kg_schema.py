import re
from typing import Dict, List


ALLOWED_NODE_TYPES = {
    "Claim",
    "EvidenceSpan",
    "Experiment",
    "PriorLimitation",
    "ResearchProblem",
    "Contribution",
    "Method",
    "MethodComponent",
    "Dataset",
    "Benchmark",
    "Metric",
    "Model",
    "Baseline",
    "ExperimentSetting",
    "Result",
    "Finding",
    "AblationFactor",
    "TheoreticalClaim",
    "TheoreticalGuarantee",
    "Limitation",
    "FutureWork",
    "CodeResource",
    "Funding",
    "Table",
    "Figure",
}

ALLOWED_EDGE_TYPES = {
    "addresses",
    "claims",
    "proposes",
    "has_component",
    "evaluated_by",
    "supported_by",
    "reported_in",
    "derived_from",
    "narrows_scope_of",
    "uses_dataset",
    "evaluates_on",
    "uses_metric",
    "uses_model",
    "compares_with",
    "improves_over",
    "reports_result",
    "supports",
    "has_ablation",
    "has_limitation",
    "suggests_future_work",
    "funded_by",
    "has_resource",
    "mentions",
}

NODE_LIMITS = {
    "Claim": 6,
    "EvidenceSpan": 10,
    "Experiment": 6,
    "PriorLimitation": 5,
    "Method": 3,
    "MethodComponent": 8,
    "Dataset": 8,
    "Benchmark": 8,
    "Metric": 8,
    "Model": 8,
    "Baseline": 8,
    "Contribution": 5,
    "Result": 10,
    "Finding": 8,
    "Limitation": 5,
    "FutureWork": 5,
}

GENERIC_NAMES = {
    "deep learning",
    "machine learning",
    "artificial intelligence",
    "experiment",
    "experiments",
    "model",
    "models",
    "method",
    "methods",
    "performance",
    "dataset",
    "metric",
    "this issue",
    "this problem",
    "this challenge",
    "the problem",
    "the challenge",
}


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def short_evidence(text: str, max_words: int = 30) -> str:
    words = compact_whitespace(text).split()
    return " ".join(words[:max_words])


def canonical_key(node: Dict) -> str:
    return f"{node.get('type', '').lower()}::{normalize_label(node.get('name', ''))}"


def normalize_label(label: str) -> str:
    value = compact_whitespace(label).lower()
    value = value.replace("halluedit", "hulluedit")
    value = re.sub(r"[^a-z0-9.+-]+", " ", value)
    return compact_whitespace(value)


def empty_kg() -> Dict:
    return {"nodes": [], "edges": []}


def section_focus(section_name: str) -> str:
    name = compact_whitespace(section_name).lower()
    if "abstract" in name or "introduction" in name:
        return "problem_claim_contribution"
    if any(token in name for token in ("method", "approach", "model", "framework")):
        return "method_mechanism"
    if any(token in name for token in ("experiment", "evaluation", "result", "analysis", "ablation", "benchmark")):
        return "experiment_evidence"
    if any(token in name for token in ("conclusion", "limitation", "future")):
        return "scope_limitation_future"
    if "acknowledg" in name:
        return "funding"
    return "general"


def validate_and_compact_kg(raw: Dict, section_name: str) -> Dict:
    if not isinstance(raw, dict):
        return empty_kg()

    nodes = []
    seen = set()
    counts = {}
    id_map = {}

    for idx, item in enumerate(raw.get("nodes", [])):
        if not isinstance(item, dict):
            continue
        node_type = compact_whitespace(item.get("type"))
        name = compact_whitespace(item.get("name"))
        if node_type not in ALLOWED_NODE_TYPES or not name:
            continue
        if name.lower() in GENERIC_NAMES:
            continue

        limit = NODE_LIMITS.get(node_type, 5)
        if counts.get(node_type, 0) >= limit:
            continue

        node_id = compact_whitespace(item.get("id")) or f"{node_type.lower()}_{idx + 1}"
        key = f"{node_type.lower()}::{name.lower()}"
        if key in seen:
            continue
        seen.add(key)
        counts[node_type] = counts.get(node_type, 0) + 1
        id_map[node_id] = node_id

        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}

        research_value_score = _confidence(
            item.get("research_value_score", item.get("research_value", item.get("confidence")))
        )

        nodes.append({
            "id": node_id,
            "type": node_type,
            "name": name,
            "aliases": [compact_whitespace(x) for x in aliases if compact_whitespace(x) and compact_whitespace(x) != name][:5],
            "summary": compact_whitespace(item.get("summary"))[:300],
            "evidence": {
                "section": compact_whitespace(evidence.get("section")) or section_name,
                "text": short_evidence(evidence.get("text")),
            },
            "confidence": _confidence(item.get("confidence")),
            "research_value_score": research_value_score,
        })

    node_ids = {n["id"] for n in nodes}
    edges = []
    edge_seen = set()
    for item in raw.get("edges", []):
        if not isinstance(item, dict):
            continue
        source = compact_whitespace(item.get("source"))
        target = compact_whitespace(item.get("target"))
        edge_type = compact_whitespace(item.get("type"))
        if source not in node_ids or target not in node_ids or edge_type not in ALLOWED_EDGE_TYPES:
            continue
        key = (source, target, edge_type)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        edges.append({
            "source": source,
            "target": target,
            "type": edge_type,
            "evidence": {
                "section": compact_whitespace(evidence.get("section")) or section_name,
                "text": short_evidence(evidence.get("text")),
            },
            "confidence": _confidence(item.get("confidence")),
        })

    return {"nodes": nodes, "edges": edges}


def merge_section_kgs(section_kgs: List[Dict]) -> Dict:
    merged_nodes = []
    merged_edges = []
    key_to_id = {}
    section_id_maps = []
    global_counts = {}

    for kg in section_kgs:
        old_to_new = {}
        for node in kg.get("nodes", []):
            key = canonical_key(node)
            if key in key_to_id:
                old_to_new[node["id"]] = key_to_id[key]
                existing = next(n for n in merged_nodes if n["id"] == key_to_id[key])
                for alias in node.get("aliases", []):
                    if alias not in existing["aliases"] and alias != existing["name"]:
                        existing["aliases"].append(alias)
                continue
            new_id = f"n{len(merged_nodes) + 1}"
            node_type = node.get("type", "")
            global_limit = NODE_LIMITS.get(node_type, 5)
            if global_counts.get(node_type, 0) >= global_limit:
                continue
            global_counts[node_type] = global_counts.get(node_type, 0) + 1
            old_to_new[node["id"]] = new_id
            copied = dict(node)
            copied["id"] = new_id
            key_to_id[key] = new_id
            merged_nodes.append(copied)
        section_id_maps.append(old_to_new)

    edge_seen = set()
    for kg, old_to_new in zip(section_kgs, section_id_maps):
        for edge in kg.get("edges", []):
            source = old_to_new.get(edge.get("source"))
            target = old_to_new.get(edge.get("target"))
            if not source or not target:
                continue
            key = (source, target, edge.get("type"))
            if key in edge_seen:
                continue
            edge_seen.add(key)
            copied = dict(edge)
            copied["source"] = source
            copied["target"] = target
            merged_edges.append(copied)

    return enrich_graph_links({"nodes": merged_nodes, "edges": merged_edges})


def enrich_graph_links(kg: Dict) -> Dict:
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    if not nodes:
        return kg

    method_nodes = [n for n in nodes if n.get("type") == "Method"]
    if not method_nodes:
        return kg
    main_method = sorted(
        method_nodes,
        key=lambda n: (float(n.get("research_value_score", 0.0)), float(n.get("confidence", 0.0))),
        reverse=True,
    )[0]

    connected = set()
    edge_seen = {(e.get("source"), e.get("target"), e.get("type")) for e in edges}
    for edge in edges:
        connected.add(edge.get("source"))
        connected.add(edge.get("target"))

    for node in nodes:
        node_id = node.get("id")
        if node_id in connected or node_id == main_method.get("id"):
            continue
        node_type = node.get("type")
        rel_type = None
        if node_type in ("ResearchProblem", "PriorLimitation", "Limitation"):
            rel_type = "addresses"
        elif node_type == "Model":
            rel_type = "uses_model"
        elif node_type == "Dataset":
            rel_type = "uses_dataset"
        elif node_type == "Baseline":
            rel_type = "compares_with"
        elif node_type == "Benchmark":
            rel_type = "evaluates_on"

        if not rel_type:
            continue
        key = (main_method["id"], node_id, rel_type)
        if key in edge_seen:
            continue
        evidence = node.get("evidence", {}) if isinstance(node.get("evidence"), dict) else {}
        edges.append({
            "source": main_method["id"],
            "target": node_id,
            "type": rel_type,
            "evidence": {
                "section": evidence.get("section", ""),
                "text": evidence.get("text", ""),
            },
            "confidence": min(0.55, float(node.get("confidence", 0.0) or 0.0)),
            "inferred_by_postprocess": True,
        })
        connected.add(main_method["id"])
        connected.add(node_id)
        edge_seen.add(key)

    kg["edges"] = edges
    return kg


def build_quality_report(kg: Dict) -> Dict:
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    node_ids = {n.get("id") for n in nodes}
    connected = set()
    for edge in edges:
        connected.add(edge.get("source"))
        connected.add(edge.get("target"))

    low_confidence = [
        n.get("id") for n in nodes
        if float(n.get("confidence", 0.0)) < 0.5 or float(n.get("research_value_score", 0.0)) < 0.5
    ]
    missing_evidence = [
        n.get("id") for n in nodes
        if not compact_whitespace((n.get("evidence") or {}).get("text", ""))
    ]
    isolated = sorted(node_ids - connected)
    by_type = {}
    for node in nodes:
        by_type[node.get("type", "")] = by_type.get(node.get("type", ""), 0) + 1

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_type": by_type,
        "isolated_node_count": len(isolated),
        "isolated_nodes": isolated[:20],
        "low_confidence_nodes": low_confidence[:20],
        "missing_evidence_nodes": missing_evidence[:20],
        "claim_count": by_type.get("Claim", 0),
        "evidence_count": by_type.get("EvidenceSpan", 0),
        "result_count": by_type.get("Result", 0),
    }


def _confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
