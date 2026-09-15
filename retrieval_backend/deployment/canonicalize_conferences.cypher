// Merge case-only legacy conference duplicates into the rich canonical nodes.
UNWIND [
  {legacy: 'conference:legacy:3723b18df05a80312ecc623b', canonical: 'conference:icml_2026'},
  {legacy: 'conference:legacy:cfbd912403a11215d4bb2cd8', canonical: 'conference:s_p_2026'},
  {legacy: 'conference:legacy:0cf7c36a74cbe9ac61826dd2', canonical: 'conference:vldb_2026'}
] AS mapping
MATCH (legacy:Conference {conference_id: mapping.legacy})
MATCH (canonical:Conference {conference_id: mapping.canonical})
CALL {
  WITH legacy, canonical
  OPTIONAL MATCH (paper:Paper)-[old:PUBLISHED_IN]->(legacy)
  WITH canonical, paper, old WHERE paper IS NOT NULL
  MERGE (paper)-[replacement:PUBLISHED_IN]->(canonical)
  ON CREATE SET replacement = properties(old)
  DELETE old
}
CALL {
  WITH legacy, canonical
  OPTIONAL MATCH (legacy)-[old:PART_OF]->(venue:Venue)
  WITH canonical, venue, old WHERE venue IS NOT NULL
  MERGE (canonical)-[replacement:PART_OF]->(venue)
  ON CREATE SET replacement = properties(old)
  DELETE old
}
DELETE legacy;
