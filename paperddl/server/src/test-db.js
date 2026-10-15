import "./env.js";
import mysql from "mysql2/promise";
import Database from "better-sqlite3";
import path from "node:path";

const configs = [
  {
    name: "user",
    host: process.env.USER_MYSQL_HOST || process.env.MYSQL_HOST,
    port: Number(process.env.USER_MYSQL_PORT || process.env.MYSQL_PORT || 3306),
    user: process.env.USER_MYSQL_USER || process.env.MYSQL_USER,
    password: process.env.USER_MYSQL_PASSWORD ?? process.env.MYSQL_PASSWORD,
    database: process.env.USER_MYSQL_DATABASE || process.env.MYSQL_DATABASE || "paperddl"
  },
];

for (const config of configs) {
  const { name, ...connectionConfig } = config;
  console.log(`[db-check] ${name}: ${config.host}:${config.port}/${config.database}`);
  try {
    const connection = await mysql.createConnection({ ...connectionConfig, connectTimeout: 5000 });
    const [rows] = await connection.execute("SELECT VERSION() AS version");
    console.log(`[db-check] ${name}: connected (${rows[0].version})`);
    await connection.end();
  } catch (error) {
    console.error(`[db-check] ${name}: failed: ${error.message}`);
    process.exitCode = 1;
  }
}

const sqlitePath = path.resolve(
  process.env.PAPER_SQLITE_PATH || "/home/user2/jiansuo/retrieval_backend/papers_fts.db"
);
console.log(`[db-check] papers: SQLite ${sqlitePath}`);
try {
  const db = new Database(sqlitePath, { readonly: true, fileMustExist: true });
  const rows = db.prepare(
    "SELECT record_type, COUNT(*) AS total FROM papers GROUP BY record_type ORDER BY record_type"
  ).all();
  console.log(`[db-check] papers: SQLite connected (${rows.map((row) => `${row.record_type}=${row.total}`).join(", ")})`);
  db.close();
} catch (error) {
  console.error(`[db-check] papers: SQLite failed: ${error.message}`);
  process.exitCode = 1;
}

if (process.env.CHECK_PAPER_MYSQL === "1") {
  const config = {
    name: "legacy-paper-mysql",
    host: process.env.PAPER_MYSQL_HOST || process.env.MYSQL_HOST,
    port: Number(process.env.PAPER_MYSQL_PORT || 3306),
    user: process.env.PAPER_MYSQL_USER || process.env.MYSQL_USER,
    password: process.env.PAPER_MYSQL_PASSWORD ?? process.env.MYSQL_PASSWORD,
    database: process.env.PAPER_MYSQL_DATABASE || "paperddl_papers"
  };
  const { name, ...connectionConfig } = config;
  console.log(`[db-check] ${name}: ${config.host}:${config.port}/${config.database}`);
  try {
    const connection = await mysql.createConnection({ ...connectionConfig, connectTimeout: 5000 });
    const [rows] = await connection.execute("SELECT VERSION() AS version");
    console.log(`[db-check] ${name}: connected (${rows[0].version})`);
    await connection.end();
  } catch (error) {
    console.error(`[db-check] ${name}: failed: ${error.message}`);
    process.exitCode = 1;
  }
}
