import "./env.js";
import mysql from "mysql2/promise";

const port = Number(process.env.USER_MYSQL_PORT || process.env.MYSQL_PORT || 3306);
const connectionLimit = Number(process.env.USER_MYSQL_CONNECTION_LIMIT || process.env.MYSQL_CONNECTION_LIMIT || 10);

// User database: accounts, roles, sessions-related fields, and favorites.
export const pool = mysql.createPool({
  host: process.env.USER_MYSQL_HOST || process.env.MYSQL_HOST || "localhost",
  port,
  user: process.env.USER_MYSQL_USER || process.env.MYSQL_USER || "root",
  password: process.env.USER_MYSQL_PASSWORD ?? process.env.MYSQL_PASSWORD ?? "",
  database: process.env.USER_MYSQL_DATABASE || process.env.MYSQL_DATABASE || "paperddl",
  waitForConnections: true,
  connectionLimit,
  queueLimit: 0
});

export async function execute(sql, params = []) {
  const [rows] = await pool.execute(sql, params);
  return rows;
}
