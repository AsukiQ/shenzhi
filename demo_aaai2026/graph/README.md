# AAAI 2026 demo graph

This directory is a scoped export of the 5,123-paper AAAI 2026 demo set.

Counts: 5,123 papers, 19,290 authors, 13,223 topics, 889 institutions, 1 conference, and 85,290 relationships.

The export is intentionally separate from the production Neo4j store. Import it into an empty Neo4j database with the CSV files in this directory. `properties_json` contains the original node properties and can be unpacked by an import adapter if needed. The current source graph has no AAAI-to-AAAI `CITES` or `FUNDED_BY` relationships, so those are not fabricated here.
