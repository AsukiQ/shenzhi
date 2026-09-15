import { once } from "node:events";
import readline from "node:readline";

const excludedIds = new Set(process.argv.slice(2));
const seenIds = new Set();
const stats = { input: 0, output: 0, excluded: 0 };

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line.trim()) continue;
  stats.input += 1;
  const document = JSON.parse(line);
  const paperId = String(document.paper_id || "").trim();
  if (!paperId) throw new Error(`retrieval document ${stats.input} has no paper_id`);
  if (seenIds.has(paperId)) throw new Error(`duplicate retrieval paper_id: ${paperId}`);
  seenIds.add(paperId);
  if (excludedIds.has(paperId)) {
    stats.excluded += 1;
    continue;
  }
  if (!process.stdout.write(`${line}\n`)) await once(process.stdout, "drain");
  stats.output += 1;
}

console.error(JSON.stringify(stats));
