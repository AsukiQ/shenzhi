CREATE INDEX rich_paper_external_year IF NOT EXISTS
FOR (p:Paper) ON (p.external, p.year);

CREATE FULLTEXT INDEX paper_title_fulltext IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title];
