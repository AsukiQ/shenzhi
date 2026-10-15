import "./env.js";
import mysql from "mysql2/promise";

const configs = [
  {
    name: "user",
    host: process.env.USER_MYSQL_HOST || process.env.MYSQL_HOST,
    port: Number(process.env.USER_MYSQL_PORT || process.env.MYSQL_PORT || 3306),
    user: process.env.USER_MYSQL_USER || process.env.MYSQL_USER,
    password: process.env.USER_MYSQL_PASSWORD ?? process.env.MYSQL_PASSWORD,
    database: process.env.USER_MYSQL_DATABASE || process.env.MYSQL_DATABASE || "paperddl"
  },
  {
    name: "paper",
    host: process.env.PAPER_MYSQL_HOST || process.env.MYSQL_HOST,
    port: Number(process.env.PAPER_MYSQL_PORT || 3306),
    user: process.env.PAPER_MYSQL_USER || process.env.MYSQL_USER,
    password: process.env.PAPER_MYSQL_PASSWORD ?? process.env.MYSQL_PASSWORD,
    database: process.env.PAPER_MYSQL_DATABASE || "paperddl_papers"
  }
];

for (const config of configs) {
  console.log(`[db-check] ${config.name}: ${config.host}:${config.port}/${config.database}`);
  try {
    const connection = await mysql.createConnection({ ...config, connectTimeout: 5000 });
    const [rows] = await connection.execute("SELECT VERSION() AS version");
    console.log(`[db-check] ${config.name}: connected (${rows[0].version})`);
    await connection.end();
  } catch (error) {
    console.error(`[db-check] ${config.name}: failed: ${error.message}`);
    process.exitCode = 1;
  }
}
