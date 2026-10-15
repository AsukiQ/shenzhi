/** Consistent InnoDB logical export; env-file contains MYSQL_* settings. */
import fs from 'node:fs';
import {once} from 'node:events';
import {createRequire} from 'node:module';
const require = createRequire(new URL('../paperddl/server/package.json',import.meta.url));
const mysql = require('mysql2/promise');
const dotenv = require('dotenv');
const [envFile, target] = process.argv.slice(2);
if (!envFile || !target) throw Error('Usage: node export-mysql.mjs /path/to/server.env /path/to/mysql.sql');
const cfg={...dotenv.parse(fs.readFileSync(envFile)),...process.env};
const conn=await mysql.createConnection({host:cfg.USER_MYSQL_HOST||cfg.MYSQL_HOST,port:Number(cfg.USER_MYSQL_PORT||cfg.MYSQL_PORT||3306),user:cfg.USER_MYSQL_USER||cfg.MYSQL_USER,password:cfg.USER_MYSQL_PASSWORD??cfg.MYSQL_PASSWORD,database:cfg.USER_MYSQL_DATABASE||cfg.MYSQL_DATABASE||'paperddl',supportBigNumbers:true,bigNumberStrings:true,dateStrings:true,jsonStrings:true,connectTimeout:20000});
const stream=fs.createWriteStream(target,{flags:'wx',mode:0o600});
const write=async value=>{if(!stream.write(value))await once(stream,'drain');};
const ident=value=>'`'+value.replaceAll('`','``')+'`';
const counts={};
try {
 const [tables]=await conn.query('SHOW TABLE STATUS');
 if(tables.some(t=>t.Engine!=='InnoDB')) throw Error('Export requires all tables to be InnoDB');
 await conn.query('SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ');
 await conn.query('START TRANSACTION WITH CONSISTENT SNAPSHOT');
 await write('-- Shenzhi consistent MySQL export; import into an empty database.\nSET NAMES utf8mb4;\nSET FOREIGN_KEY_CHECKS=0;\nSET @OLD_SQL_MODE=@@SQL_MODE;\nSET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";\n');
 for(const t of tables){
  const name=t.Name;
  const [ddl]=await conn.query('SHOW CREATE TABLE '+ident(name));
  await write('\n'+ddl[0]['Create Table']+';\n');
  const [columns]=await conn.query('SHOW COLUMNS FROM '+ident(name));
  const names=columns.filter(c=>!String(c.Extra).includes('GENERATED')).map(c=>c.Field);
  let offset=0;
  const order=columns.filter(c=>c.Key==='PRI').map(c=>ident(c.Field)).join(',');
  for(;;){
   const [rows]=await conn.query('SELECT '+names.map(ident).join(',')+' FROM '+ident(name)+(order?' ORDER BY '+order:'')+' LIMIT 2000 OFFSET '+offset);
   if(!rows.length)break;
   for(const row of rows)await write('INSERT INTO '+ident(name)+' ('+names.map(ident).join(',')+') VALUES ('+names.map(k=>conn.escape(row[k])).join(',')+');\n');
   offset+=rows.length;
   if(offset%10000===0)console.log(name,offset);
  }
  counts[name]=offset;console.log('exported',name,offset);
 }
 const [triggers]=await conn.query('SHOW TRIGGERS');
 const [routines]=await conn.query('SHOW PROCEDURE STATUS WHERE Db=DATABASE()');
 const [functions]=await conn.query('SHOW FUNCTION STATUS WHERE Db=DATABASE()');
 const [events]=await conn.query('SHOW EVENTS');
 if(triggers.length||routines.length||functions.length||events.length)throw Error('Database has triggers/routines/events; native mysqldump required');
 await write('SET SQL_MODE=@OLD_SQL_MODE;\nSET FOREIGN_KEY_CHECKS=1;\n');
 await conn.commit();
 stream.end();await once(stream,'finish');
 fs.writeFileSync(target+'.manifest.json',JSON.stringify({created_at:new Date().toISOString(),database:cfg.USER_MYSQL_DATABASE||cfg.MYSQL_DATABASE||'paperddl',counts,consistent_snapshot:true,triggers:0,routines:0,events:0},null,2),{mode:0o600});
 console.log('export complete',counts);
} catch(error){await conn.rollback();stream.destroy();throw error;} finally{await conn.end();}
