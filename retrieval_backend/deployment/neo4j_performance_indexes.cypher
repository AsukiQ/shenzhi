CREATE INDEX rich_paper_external_year IF NOT EXISTS
FOR (p:Paper) ON (p.external, p.year);

// Candidate filtering always narrows by paper_id and then by record_type.
// The composite index keeps this path selective after the graph was expanded
// with external and reference records.
CREATE INDEX rich_paper_record_type_paper_id IF NOT EXISTS
FOR (p:Paper) ON (p.record_type, p.paper_id);

CREATE INDEX rich_paper_record_type_year IF NOT EXISTS
FOR (p:Paper) ON (p.record_type, p.year);

CREATE FULLTEXT INDEX paper_title_fulltext IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title];

// Entity search uses these indexes when a query is present.  The API keeps a
// CONTAINS fallback for legacy data and for values that are not tokenized by
// the fulltext analyzer.
CREATE FULLTEXT INDEX author_name_fulltext IF NOT EXISTS
FOR (a:Author) ON EACH [a.name, a.author_id];

CREATE FULLTEXT INDEX institution_name_fulltext IF NOT EXISTS
FOR (i:Institution) ON EACH [i.name, i.institution_id];

CREATE FULLTEXT INDEX funding_name_fulltext IF NOT EXISTS
FOR (f:Funding) ON EACH [f.name, f.funding_id];

CREATE FULLTEXT INDEX topic_name_fulltext IF NOT EXISTS
FOR (t:Topic) ON EACH [t.name, t.topic_id];
