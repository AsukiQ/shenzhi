import { paperPrepare } from "./paperDb.js";
import { toPaper } from "./paperRepository.js";

function page(value, fallback, maximum) {
  const n = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(n) || n < (fallback === 0 ? 0 : 1) || n > maximum) {
    const error = new Error("Invalid pagination"); error.status = 400; throw error;
  }
  return n;
}
const topicIds = `SELECT ps.paper_id FROM paper_subjects ps
  WHERE ps.value_key IN (SELECT alias_key FROM topic_aliases WHERE topic_id=t.id)
     OR ps.value_key=lower(t.subject_query)`;
const countSql = `(SELECT count(*) FROM papers p WHERE p.record_type='canonical'
  AND p.paper_id IN (${topicIds}))`;
const topicSelect = `SELECT t.*, ${countSql} AS paper_count FROM topic_catalog t`;
function topic(row) {
  return { id:row.id, imagePath:row.image_path, nameZh:row.name_zh, nameEn:row.name_en,
    descriptionZh:row.description_zh, descriptionEn:row.description_en,
    category:row.category, subjectQuery:row.subject_query, paperCount:Number(row.paper_count),
    enabled:row.enabled === 1 };
}
export function listTopics({enabled=true, category='', limit=100, offset=0}={}) {
  limit=page(limit,100,200); offset=page(offset,0,1_000_000_000);
  const clauses=[], args=[];
  if (enabled !== 'all') { clauses.push('t.enabled=?'); args.push(enabled ? 1 : 0); }
  if (category) { clauses.push('t.category=?'); args.push(category); }
  const where=clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
  const total=paperPrepare(`SELECT count(*) AS n FROM topic_catalog t ${where}`).get(...args).n;
  const rows=paperPrepare(`${topicSelect} ${where} ORDER BY t.sort_order,t.name_en,t.id LIMIT ? OFFSET ?`).all(...args,limit,offset);
  return {items:rows.map(topic),total,limit,offset};
}
export function getTopic(id) {
  const row=paperPrepare(`${topicSelect} WHERE t.id=?`).get(id);
  return row ? topic(row) : null;
}
export function listTopicPapers(id,{limit=20,offset=0}={}) {
  limit=page(limit,20,200); offset=page(offset,0,1_000_000_000);
  const total=getTopic(id)?.paperCount || 0;
  const rows=paperPrepare(`SELECT p.* FROM papers p, topic_catalog t WHERE t.id=?
    AND p.record_type='canonical' AND p.paper_id IN (${topicIds})
    ORDER BY p.year DESC,p.paper_id LIMIT ? OFFSET ?`).all(id,limit,offset);
  return {items:rows.map(toPaper),total,limit,offset};
}
export function getScholarProfile(id) {
  const profile=paperPrepare('SELECT * FROM scholar_profiles WHERE scholar_id=?').get(id) || null;
  const affiliations=paperPrepare('SELECT institution,position FROM scholar_affiliations WHERE scholar_id=? ORDER BY institution,position').all(id);
  const year=new Date().getUTCFullYear();
  const annualCitations=paperPrepare('SELECT year,count FROM scholar_citations WHERE scholar_id=? AND year BETWEEN ? AND ? ORDER BY year').all(id,year-4,year);
  const details=paperPrepare('SELECT * FROM scholar_details WHERE scholar_id=?').get(id);
  return {profile,affiliations,annualCitations,
    affiliationHistory: details ? JSON.parse(details.affiliation_history_json) : [],
    researchImpact: details ? JSON.parse(details.research_impact_json) : {}};
}
export function scholarLinkedPaperIds(id) {
  return paperPrepare('SELECT paper_id FROM scholar_paper_links WHERE scholar_id=?').all(id).map(row => row.paper_id);
}
export function searchScholarProfiles(query, {limit=20, offset=0}={}) {
  limit=page(limit,20,100); offset=page(offset,0,1_000_000_000);
  const q=String(query || '').trim().toLowerCase();
  const like=`%${q}%`;
  const rawLike=`%${String(query || '').trim()}%`;
  const rows=paperPrepare(`SELECT * FROM scholar_profiles
    WHERE lower(coalesce(scholar_id,'')) LIKE ? OR lower(coalesce(name_en,'')) LIKE ? OR lower(coalesce(name_zh,'')) LIKE ?
       OR scholar_id LIKE ? OR name_en LIKE ? OR name_zh LIKE ?
    ORDER BY name_en,scholar_id LIMIT ? OFFSET ?`).all(like,like,like,rawLike,rawLike,rawLike,limit,offset);
  return rows;
}
// JSON input keeps arbitrarily large scholar bibliographies below SQLite's bind-variable limit.
export function listScholarPapers(ids,{limit=20,offset=0}={}) {
  limit=page(limit,20,200); offset=page(offset,0,1_000_000_000);
  const json=JSON.stringify([...new Set(ids)]);
  const where="record_type IN ('canonical','external_full') AND paper_id IN (SELECT value FROM json_each(?))";
  const total=paperPrepare(`SELECT count(*) AS n FROM papers WHERE ${where}`).get(json).n;
  const rows=paperPrepare(`SELECT * FROM papers WHERE ${where} ORDER BY year DESC,paper_id LIMIT ? OFFSET ?`).all(json,limit,offset);
  return {items:rows.map(toPaper),total,limit,offset};
}
