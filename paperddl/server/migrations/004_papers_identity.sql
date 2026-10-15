-- Upgrade an existing paper database that was created before the dual-MySQL split.
-- Run only if 003_papers.sql was already applied without paper_id_hash.
SET @has_hash := (
  SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'papers' AND COLUMN_NAME = 'paper_id_hash'
);
SET @sql := IF(@has_hash = 0,
  'ALTER TABLE papers ADD COLUMN paper_id_hash BINARY(32) GENERATED ALWAYS AS (UNHEX(SHA2(paper_id, 256))) STORED, ADD UNIQUE KEY uq_papers_paper_id_hash (paper_id_hash)',
  'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
