import json
import logging
import re
from typing import Dict

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from kg_schema import empty_kg, section_focus, validate_and_compact_kg

logger = logging.getLogger(__name__)


KG_RESPONSE_SCHEMA = {
    "name": "academic_claim_evidence_kg",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["nodes", "edges"],
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "type",
                        "name",
                        "aliases",
                        "summary",
                        "evidence",
                        "confidence",
                        "research_value_score",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "evidence": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["section", "text"],
                            "properties": {
                                "section": {"type": "string"},
                                "text": {"type": "string"},
                            },
                        },
                        "confidence": {"type": "number"},
                        "research_value_score": {"type": "number"},
                    },
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "target", "type", "evidence", "confidence"],
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "type": {"type": "string"},
                        "evidence": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["section", "text"],
                            "properties": {
                                "section": {"type": "string"},
                                "text": {"type": "string"},
                            },
                        },
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
    },
    "strict": True,
}


SYSTEM_PROMPT = """You are a scientific claim-evidence knowledge graph extraction engine.

Return ONLY valid JSON. No markdown. No explanations.

Hard constraints:
1. Extract only information explicitly supported by the text.
2. Do not infer missing facts.
3. Do not output generic terms such as "deep learning", "experiment", "model", "performance" unless they are part of a specific named entity.
4. Merge duplicates and aliases inside this section.
5. Prefer fewer high-value nodes over many weak nodes.
6. Every node and edge must have evidence.
7. Evidence text must be short, maximum 30 words.
8. Use original English names for methods, datasets, metrics, benchmarks, and models.
9. If no useful information exists, return empty arrays.
10. Output must exactly match the schema.
11. Prioritize scientific reasoning: claim, evidence, scope, limitation, and reproducibility.
12. Create EvidenceSpan nodes for short text spans that support a claim, result, or limitation.
13. The top-level JSON object must contain exactly two keys: "nodes" and "edges".
14. Every edge source and target must match an id from nodes.
15. Do not create standalone weak nodes. If a node cannot be connected by a meaningful edge, omit it.
16. Evidence text must be copied from the section and must not exceed 30 words.
17. confidence and research_value_score must be numbers between 0 and 1.

Allowed node types:
Claim, EvidenceSpan, Experiment, PriorLimitation,
ResearchProblem, Contribution, Method, MethodComponent, Dataset, Benchmark,
Metric, Model, Baseline, ExperimentSetting, Result, Finding,
AblationFactor, TheoreticalClaim, TheoreticalGuarantee, Limitation,
FutureWork, CodeResource, Funding, Table, Figure.

Allowed edge types:
addresses, claims, proposes, has_component, evaluated_by, supported_by,
reported_in, derived_from, narrows_scope_of, uses_dataset, evaluates_on,
uses_metric, uses_model, compares_with, improves_over, reports_result,
supports, has_ablation, has_limitation, suggests_future_work, funded_by,
has_resource, mentions.

Maximum nodes:
- Claim: 6
- EvidenceSpan: 10
- Experiment: 6
- Method: 3
- MethodComponent: 8
- Dataset or Benchmark: 8
- Metric: 8
- Model: 8
- Baseline: 8
- Contribution: 5
- Result: 10
- Other types: 5 each

Node schema:
{
  "id": "local unique id, e.g. method_1",
  "type": "one allowed node type",
  "name": "canonical concise name",
  "aliases": [],
  "summary": "one short sentence",
  "evidence": {
    "section": "section name",
    "text": "short supporting span"
  },
  "confidence": 0.0,
  "research_value_score": 0.0
}

Edge schema:
{
  "source": "node id",
  "target": "node id",
  "type": "one allowed edge type",
  "evidence": {
    "section": "section name",
    "text": "short supporting span"
  },
  "confidence": 0.0
}

Output:
{
  "nodes": [],
  "edges": []
}

Do not output any other fields such as paper, metadata, quality_report, notes, or explanations."""


class AcademicKGExtractor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.model = LLM_MODEL
        if not dry_run and not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is required when dry_run is False.")
        self.client = None if dry_run else OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def extract_section(self, paper: Dict, section: Dict) -> Dict:
        if section_focus(section.get("title", "")) == "funding":
            funding_kg = self._extract_funding_rules(section)
            if funding_kg.get("nodes"):
                return validate_and_compact_kg(funding_kg, section.get("title", ""))
        if self.dry_run:
            return self._dry_run_extract(paper, section)

        user_prompt = self._build_user_prompt(paper, section)
        response_modes = ("plain", "json_object", "json_schema") if self.model.startswith("glm-") else (
            "json_schema",
            "json_object",
            "plain",
        )
        for response_mode in response_modes:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                }
                if response_mode == "json_schema":
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": KG_RESPONSE_SCHEMA,
                    }
                elif response_mode == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                if not raw.strip():
                    raise ValueError("empty model content")
                parsed = json.loads(self._extract_json(raw))
                return validate_and_compact_kg(parsed, section.get("title", ""))
            except Exception as exc:
                if response_mode != "plain":
                    logger.warning("KG extraction with %s failed: %s", response_mode, exc)
                    continue
                logger.warning("KG extraction with plain failed: %s", exc)
                continue
        logger.error("KG extraction failed in all response modes.")
        return empty_kg()

    def _extract_funding_rules(self, section: Dict) -> Dict:
        title = section.get("title", "")
        text = section.get("text", "")
        cleaned = re.sub(r"^This work was supported by\s+", "", text.strip(), flags=re.I)
        cleaned = re.sub(r"^This work is supported by\s+", "", cleaned, flags=re.I)
        parts = re.split(r";|\band\b the |\band\b Open Fund", cleaned)

        nodes = []
        edges = []
        seen = set()
        for part in parts:
            item = part.strip(" .")
            if not item:
                continue
            if part != item:
                pass
            if "Open Fund" not in item and "Key Laboratory" in item:
                item = "Open Fund " + item
            item = re.sub(r"^in part by\s+", "", item, flags=re.I).strip()
            item = re.sub(r"^by\s+", "", item, flags=re.I).strip()

            match = re.match(r"(.+?)\s*\((?:No\.\s*)?([^)]+)\)", item)
            if match:
                name = match.group(1).strip()
                grant_id = match.group(2).strip()
            else:
                name = item.strip()
                grant_id = ""
            name = re.sub(r"^(the|and)\s+", "", name, flags=re.I).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            idx = len(nodes) // 2 + 1
            funding_id = f"funding_{idx}"
            evidence_id = f"funding_evidence_{idx}"
            evidence_text = item
            aliases = [grant_id] if grant_id else []

            nodes.append({
                "id": funding_id,
                "type": "Funding",
                "name": name,
                "aliases": aliases,
                "summary": f"Funding source acknowledged by the paper: {name}.",
                "evidence": {"section": title, "text": evidence_text},
                "confidence": 1.0,
                "research_value_score": 0.65,
            })
            nodes.append({
                "id": evidence_id,
                "type": "EvidenceSpan",
                "name": f"Funding evidence: {name}",
                "aliases": [],
                "summary": evidence_text,
                "evidence": {"section": title, "text": evidence_text},
                "confidence": 1.0,
                "research_value_score": 0.6,
            })
            edges.append({
                "source": funding_id,
                "target": evidence_id,
                "type": "supported_by",
                "evidence": {"section": title, "text": evidence_text},
                "confidence": 1.0,
            })

        return {"nodes": nodes, "edges": edges}
        return empty_kg()

    def _build_user_prompt(self, paper: Dict, section: Dict) -> str:
        focus = section_focus(section.get("title", ""))
        focus_instruction = self._focus_instruction(focus)
        return f"""Task:
Extract a compact, non-redundant academic claim-evidence KG from the given paper section.

Section focus:
{focus_instruction}

Input:
paper_id: {paper.get("arxiv_id", "")}
paper_title: {paper.get("title", "")}
section_name: {section.get("title", "")}
section_text:
{section.get("text", "")}
"""

    def _focus_instruction(self, focus: str) -> str:
        instructions = {
            "problem_claim_contribution": (
                "Extract ResearchProblem, PriorLimitation, Contribution, Claim, Method, "
                "and EvidenceSpan. Link Claim supported_by EvidenceSpan. Link Method addresses ResearchProblem."
            ),
            "method_mechanism": (
                "Extract Method, MethodComponent, Algorithm-like mechanisms as MethodComponent, "
                "TheoreticalClaim, TheoreticalGuarantee, Claim, and EvidenceSpan. Link Method has_component MethodComponent."
            ),
            "experiment_evidence": (
                "Extract Experiment, Dataset, Benchmark, Metric, Model, Baseline, Result, Finding, "
                "Claim, EvidenceSpan, Table/Figure if explicitly referenced. Link Result supports Claim and Result reported_in EvidenceSpan/Table."
            ),
            "scope_limitation_future": (
                "Extract Claim, Limitation, FutureWork, Finding, EvidenceSpan. Link Limitation narrows_scope_of Claim."
            ),
            "funding": "Extract Funding and EvidenceSpan only.",
            "general": "Extract only high-value Claim, Method, Result, Limitation, and EvidenceSpan nodes.",
        }
        return instructions.get(focus, instructions["general"])

    def _extract_json(self, raw: str) -> str:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```", "", cleaned).strip()
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return match.group(0)
        return cleaned

    def _dry_run_extract(self, paper: Dict, section: Dict) -> Dict:
        title = section.get("title", "")
        text = section.get("text", "")
        lower = f"{title}\n{text}".lower()
        nodes = []
        edges = []
        method_id = None
        focus = section_focus(title)

        claim = self._infer_claim(text, focus)
        if claim:
            nodes.append({
                "id": "claim_1",
                "type": "Claim",
                "name": claim,
                "aliases": [],
                "summary": f"Scientific claim stated in the section: {claim}.",
                "evidence": {"section": title, "text": self._find_span(text, claim.split()[0])},
                "confidence": 0.6,
                "research_value_score": 0.75,
            })
            nodes.append({
                "id": "evidence_1",
                "type": "EvidenceSpan",
                "name": f"Evidence from {title or 'section'}",
                "aliases": [],
                "summary": self._find_span(text, claim.split()[0])[:220],
                "evidence": {"section": title, "text": self._find_span(text, claim.split()[0])},
                "confidence": 0.65,
                "research_value_score": 0.7,
            })
            edges.append({
                "source": "claim_1",
                "target": "evidence_1",
                "type": "supported_by",
                "evidence": {"section": title, "text": self._find_span(text, claim.split()[0])},
                "confidence": 0.65,
            })

        if "hulluedit" in lower or "halluedit" in lower:
            nodes.append({
                "id": "method_1",
                "type": "Method",
                "name": "HulluEdit",
                "aliases": ["HalluEdit"],
                "summary": "Single-pass reference-free subspace editing framework for mitigating LVLM hallucinations.",
                "evidence": {"section": title, "text": self._find_span(text, "HulluEdit")},
                "confidence": 0.8,
            })
            method_id = "method_1"

        if not method_id:
            candidate = self._infer_method_name(paper, text)
            if candidate:
                method_id = "method_1"
                nodes.append({
                    "id": method_id,
                    "type": "Method",
                    "name": candidate,
                    "aliases": [],
                    "summary": f"Method or system identified from the paper section: {candidate}.",
                    "evidence": {"section": title, "text": self._find_span(text, candidate)},
                    "confidence": 0.55,
                    "research_value_score": 0.65,
                })
                if any(n["id"] == "claim_1" for n in nodes):
                    edges.append({
                        "source": "claim_1",
                        "target": method_id,
                        "type": "mentions",
                        "evidence": {"section": title, "text": self._find_span(text, candidate)},
                        "confidence": 0.55,
                    })

        if not any(n["id"] == "claim_1" for n in nodes) and method_id and focus in (
            "problem_claim_contribution",
            "method_mechanism",
        ):
            method_name = next((n["name"] for n in nodes if n["id"] == method_id), "")
            if method_name:
                nodes.append({
                    "id": "claim_1",
                    "type": "Claim",
                    "name": f"The paper presents {method_name}",
                    "aliases": [],
                    "summary": f"The section presents {method_name} as a research contribution.",
                    "evidence": {"section": title, "text": self._find_span(text, method_name)},
                    "confidence": 0.55,
                    "research_value_score": 0.6,
                })
                nodes.append({
                    "id": "evidence_1",
                    "type": "EvidenceSpan",
                    "name": f"Evidence from {title or 'section'}",
                    "aliases": [],
                    "summary": self._find_span(text, method_name)[:220],
                    "evidence": {"section": title, "text": self._find_span(text, method_name)},
                    "confidence": 0.6,
                    "research_value_score": 0.6,
                })
                edges.append({
                    "source": "claim_1",
                    "target": "evidence_1",
                    "type": "supported_by",
                    "evidence": {"section": title, "text": self._find_span(text, method_name)},
                    "confidence": 0.6,
                })
                edges.append({
                    "source": "claim_1",
                    "target": method_id,
                    "type": "mentions",
                    "evidence": {"section": title, "text": self._find_span(text, method_name)},
                    "confidence": 0.55,
                })

        problem = self._infer_problem(text)
        if problem:
            nodes.append({
                "id": "problem_1",
                "type": "ResearchProblem",
                "name": problem,
                "aliases": [],
                "summary": f"Research problem discussed in the section: {problem}.",
                "evidence": {"section": title, "text": self._find_span(text, problem.split()[0])},
                "confidence": 0.55,
                "research_value_score": 0.7,
            })
            if method_id:
                edges.append({
                    "source": method_id,
                    "target": "problem_1",
                    "type": "addresses",
                    "evidence": {"section": title, "text": self._find_span(text, problem.split()[0])},
                    "confidence": 0.55,
                })

        if "visual evidence subspace" in lower:
            nodes.append({
                "id": "component_1",
                "type": "MethodComponent",
                "name": "Visual Evidence Subspace",
                "aliases": [],
                "summary": "Subspace representing visual evidence used to preserve visual grounding.",
                "evidence": {"section": title, "text": self._find_span(text, "visual evidence subspace")},
                "confidence": 0.75,
                "research_value_score": 0.75,
            })
        if method_id and any(n["id"] == "component_1" for n in nodes):
            edges.append({
                "source": method_id,
                "target": "component_1",
                "type": "has_component",
                "evidence": {"section": title, "text": self._find_span(text, "visual evidence subspace")},
                "confidence": 0.75,
            })

        component_phrases = self._infer_components(text)
        component_start = 2 if any(n["id"] == "component_1" for n in nodes) else 1
        for offset, component in enumerate(component_phrases[:4], start=component_start):
            node_id = f"component_{offset}"
            if any(n["name"].lower() == component.lower() for n in nodes):
                continue
            nodes.append({
                "id": node_id,
                "type": "MethodComponent",
                "name": component,
                "aliases": [],
                "summary": f"Method component mentioned in the section: {component}.",
                "evidence": {"section": title, "text": self._find_span(text, component)},
                "confidence": 0.5,
                "research_value_score": 0.55,
            })
            if method_id:
                edges.append({
                    "source": method_id,
                    "target": node_id,
                    "type": "has_component",
                    "evidence": {"section": title, "text": self._find_span(text, component)},
                    "confidence": 0.5,
                })

        for benchmark in self._infer_benchmarks(text):
            if benchmark.lower() in lower:
                node_id = f"benchmark_{len(nodes) + 1}"
                nodes.append({
                    "id": node_id,
                    "type": "Benchmark",
                    "name": benchmark,
                    "aliases": [],
                    "summary": f"{benchmark} benchmark used in evaluation.",
                    "evidence": {"section": title, "text": self._find_span(text, benchmark)},
                    "confidence": 0.7,
                    "research_value_score": 0.65,
                })
                if method_id:
                    edges.append({
                        "source": method_id,
                        "target": node_id,
                        "type": "evaluates_on",
                        "evidence": {"section": title, "text": self._find_span(text, benchmark)},
                        "confidence": 0.65,
                    })
        result = self._infer_result(text)
        if result and focus == "experiment_evidence":
            nodes.append({
                "id": "result_1",
                "type": "Result",
                "name": result,
                "aliases": [],
                "summary": f"Experimental result reported in the section: {result}.",
                "evidence": {"section": title, "text": self._find_span(text, result.split()[0])},
                "confidence": 0.55,
                "research_value_score": 0.7,
            })
            if any(n["id"] == "claim_1" for n in nodes):
                edges.append({
                    "source": "result_1",
                    "target": "claim_1",
                    "type": "supports",
                    "evidence": {"section": title, "text": self._find_span(text, result.split()[0])},
                    "confidence": 0.55,
                })
            elif method_id:
                edges.append({
                    "source": method_id,
                    "target": "result_1",
                    "type": "reports_result",
                    "evidence": {"section": title, "text": self._find_span(text, result.split()[0])},
                    "confidence": 0.55,
                })
        return validate_and_compact_kg({"nodes": nodes, "edges": edges}, title)

    def _infer_claim(self, text: str, focus: str) -> str:
        patterns = [
            r"\bwe\s+(?:show|demonstrate|find|observe|prove|claim)\s+that\s+([^.;]{20,180})",
            r"\bour\s+(?:method|approach|framework|system)\s+([^.;]{20,160})",
            r"\bresults?\s+(?:show|demonstrate|indicate)\s+that\s+([^.;]{20,180})",
        ]
        if focus == "scope_limitation_future":
            patterns.insert(0, r"\b(?:limitation|future work|we plan|we will)\s+([^.;]{20,160})")
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                candidate = match.group(1).strip(" .,:;()")[:180]
                if self._valid_claim(candidate):
                    return candidate
        return ""

    def _valid_claim(self, candidate: str) -> bool:
        text = candidate.lower().strip()
        if len(text.split()) < 5:
            return False
        reject_prefixes = (
            "is illustrated",
            "are illustrated",
            "using ",
            "in preserving",
            "shown in",
            "as shown",
            "as illustrated",
            "the following",
        )
        if text.startswith(reject_prefixes):
            return False
        reject_terms = ("figure", "table", "pytorch", "library", "libraries", "implementation")
        if any(term in text for term in reject_terms) and not any(term in text for term in ("outperform", "improve", "achiev")):
            return False
        signal_terms = (
            "achiev",
            "address",
            "demonstrat",
            "enable",
            "improv",
            "mitigat",
            "outperform",
            "preserv",
            "propose",
            "reduce",
            "robust",
            "support",
        )
        return any(term in text for term in signal_terms)

    def _infer_method_name(self, paper: Dict, text: str) -> str:
        paper_title = paper.get("title", "")
        if ":" in paper_title:
            prefix = paper_title.split(":", 1)[0].strip()
            suffix = paper_title.split(":", 1)[1].strip()
            if not prefix.lower().startswith(("technical report", "submission")) and (
                1 <= len(prefix.split()) <= 8
            ):
                return prefix
            if prefix.lower().startswith("technical report") and suffix:
                return self._truncate_words(re.split(r"[.:]", suffix)[0].strip(), 90)

        through_match = re.search(r"\bthrough\s+([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){1,5})", paper_title)
        if through_match:
            candidate = through_match.group(1).strip()
            if candidate.lower() not in {"generative adversarial networks", "deep neural networks"}:
                return candidate

        patterns = [
            r"\bwe\s+(?:propose|introduce|present|develop)\s+(?:a|an|the)?\s*([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,7})",
            r"\bour\s+([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,7})\s+(?:method|framework|model|system|approach)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip(" .,:;()")
                if candidate and candidate.lower() not in ("method", "framework", "model"):
                    return candidate
        return ""

    def _truncate_words(self, text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0].strip()

    def _infer_problem(self, text: str) -> str:
        patterns = [
            r"\b(?:problem|challenge|task)\s+of\s+([A-Za-z][A-Za-z0-9\-\s]{5,80})",
            r"\bto\s+(?:address|solve|tackle|mitigate)\s+([A-Za-z][A-Za-z0-9\-\s]{5,80})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                candidate = re.split(r"[.,;:]", match.group(1).strip())[0][:90]
                if candidate.lower() in {"this issue", "this problem", "this challenge", "the issue"}:
                    continue
                return candidate
        return ""

    def _infer_components(self, text: str) -> list:
        phrases = []
        generic = {"In our", "The overall", "Details We", "Area Detection", "Deep neural", "Generative Adversarial"}
        patterns = [
            r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){1,5})\s+(?:module|head|loss|branch|network|framework)",
            r"\b(multi-level knowledge distillation|dynamic self-supervised learning|SD map pre-training|LDTR head|auxiliary tasks)\b",
        ]
        for idx, pattern in enumerate(patterns):
            flags = re.I if idx == 1 else 0
            for match in re.finditer(pattern, text, flags=flags):
                phrase = match.group(1).strip(" .,:;()")
                if phrase in generic:
                    continue
                if 2 <= len(phrase.split()) <= 8 and phrase.lower() not in {p.lower() for p in phrases}:
                    phrases.append(phrase)
        return phrases

    def _infer_benchmarks(self, text: str) -> list:
        known = {"POPE", "CHAIR", "MME", "ImageNet-1K", "MSCOCO", "MOSE", "MeViS", "OLUS", "MNIST", "CIFAR10", "CIFAR100", "SPECS", "FLAWS"}
        ignore = {"AAAI", "CVPR", "ICML", "HTTP", "PDF", "HTML", "GPU", "API", "LLM", "ELLIS", "TDL"}
        acronyms = set(re.findall(r"\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?\b", text))
        candidates = []
        for item in sorted(known | acronyms):
            if item in ignore or item not in text:
                continue
            if re.match(r"^(AAAI|CVPR|ICML|ICLR|NeurIPS)-?\d+", item):
                continue
            if item in known:
                candidates.append(item)
                continue
            idx = text.find(item)
            window = text[max(0, idx - 80): idx + len(item) + 80].lower()
            if any(token in window for token in ("benchmark", "dataset", "metric", "evaluation", "challenge", "score")):
                candidates.append(item)
        return candidates[:8]

    def _infer_result(self, text: str) -> str:
        patterns = [
            r"\b(?:outperforms?|improves?|achieves?)\s+[^.;]{10,140}",
            r"\b\d+(?:\.\d+)?\s*%[^.;]{0,120}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                candidate = match.group(0).strip(" .,:;()")[:160]
                window = self._find_span(text, candidate.split()[0]).lower()
                if any(token in window for token in ("result", "accuracy", "score", "benchmark", "dataset", "outperform", "improve", "%")):
                    return candidate
        return ""

    def _find_span(self, text: str, phrase: str) -> str:
        if not text:
            return phrase
        idx = text.lower().find(phrase.lower())
        if idx < 0:
            return text[:180]
        return text[max(0, idx - 60): idx + len(phrase) + 120]
