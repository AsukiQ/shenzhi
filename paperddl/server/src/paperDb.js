import "./env.js";
import mysql from "mysql2/promise";

const port = Number(process.env.PAPER_MYSQL_PORT || 3306);
const connectionLimit = Number(process.env.PAPER_MYSQL_CONNECTION_LIMIT || 16);

// Paper database: paper metadata and full-text search only.
export const paperPool = mysql.createPool({
  host: process.env.PAPER_MYSQL_HOST || process.env.MYSQL_HOST || "localhost",
  port,
  user: process.env.PAPER_MYSQL_USER || process.env.MYSQL_USER || "root",
  password: process.env.PAPER_MYSQL_PASSWORD ?? process.env.MYSQL_PASSWORD ?? "",
  database: process.env.PAPER_MYSQL_DATABASE || "paperddl_papers",
  waitForConnections: true,
  connectionLimit,
  queueLimit: 0
});

export async function paperExecute(sql, params = []) {
  const [rows] = await paperPool.execute(sql, params);
  return rows;
}
