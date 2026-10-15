import dotenv from "dotenv";
import mysql from "mysql2/promise";
import path from "path";
import { fileURLToPath } from "url";

// 加载 .env 配置文件
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, "../.env") });

async function testConnection() {
  console.log("正在尝试连接数据库...");
  console.log(`Host: ${process.env.MYSQL_HOST}`);
  console.log(`User: ${process.env.MYSQL_USER}`);
  console.log(`Database: ${process.env.MYSQL_DATABASE}`);

  try {
    const connection = await mysql.createConnection({
      host: process.env.MYSQL_HOST,
      port: Number(process.env.MYSQL_PORT || 3306),
      user: process.env.MYSQL_USER,
      password: process.env.MYSQL_PASSWORD,
      database: process.env.MYSQL_DATABASE,
      connectTimeout: 5000 // 5秒超时
    });

    console.log("✅ 数据库连接成功！");

    const [rows] = await connection.execute("SELECT VERSION() as version");
    console.log("📦 数据库版本:", rows[0].version);

    await connection.end();
    console.log("👋 已安全关闭连接。");
  } catch (error) {
    console.error("❌ 数据库连接失败:");
    console.error(`错误详情: ${error.message}`);

    if (error.code === "ETIMEDOUT") {
      console.error("提示: 连接超时，请检查阿里云 RDS 白名单是否已允许当前 IP。");
    } else if (error.code === "ER_ACCESS_DENIED_ERROR") {
      console.error("提示: 账号或密码错误，请检查 .env 文件。");
    }
  }
}

testConnection();
