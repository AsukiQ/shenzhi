// Idempotent high-confidence remediation for the live final graph.
// Preserve the bad DOI value for audit, then remove it from identity metadata.
MATCH (p:Paper)
WHERE toLower(trim(toString(coalesce(p.doi, "")))) = "10.18653/v1"
SET p.doi_quarantined = coalesce(p.doi_quarantined, p.doi),
    p.doi = "",
    p.quality_flags = CASE
      WHEN p.quality_flags IS NULL THEN ["invalid_doi_prefix"]
      WHEN NOT "invalid_doi_prefix" IN p.quality_flags THEN p.quality_flags + ["invalid_doi_prefix"]
      ELSE p.quality_flags
    END
RETURN count(p) AS invalid_doi_papers_remediated;

// Keep the nodes and relationships, but remove section fragments from normal search/retrieval.
MATCH (p:Paper)
WHERE coalesce(p.external, false) = false
  AND trim(coalesce(p.title, "")) = "1 Introduction"
SET p.external = true,
    p.quarantine_reason = coalesce(p.quarantine_reason, "section_heading_title"),
    p.quality_flags = CASE
      WHEN p.quality_flags IS NULL THEN ["section_heading_title"]
      WHEN NOT "section_heading_title" IN p.quality_flags THEN p.quality_flags + ["section_heading_title"]
      ELSE p.quality_flags
    END
RETURN count(p) AS section_heading_papers_quarantined;
