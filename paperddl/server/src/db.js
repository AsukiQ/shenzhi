import "./env.js";
import mysql from "mysql2/promise";

const port = Number(process.env.MYSQL_PORT || 3306);
const connectionLimit = Number(process.env.MYSQL_CONNECTION_LIMIT || 10);

export const pool = mysql.createPool({
	host: process.env.MYSQL_HOST || "localhost",
	port,
	user: process.env.MYSQL_USER || "root",
	password: process.env.MYSQL_PASSWORD || "",
	database: process.env.MYSQL_DATABASE || "paperddl",
	waitForConnections: true,
	connectionLimit,
	queueLimit: 0
});

export async function execute(sql, params = []) {
	const [rows] = await pool.execute(sql, params);
	return rows;
}
