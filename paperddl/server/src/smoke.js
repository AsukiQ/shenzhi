import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadCategories, loadConferences } from "./repository.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const serverRoot = path.resolve(__dirname, "..");

const dataRoot = process.env.DATA_ROOT
	? path.resolve(serverRoot, process.env.DATA_ROOT)
	: path.resolve(serverRoot, "../conference");

async function main() {
	const categories = await loadCategories(dataRoot);
	const conferences = await loadConferences(dataRoot);
	console.log(`[smoke] data root: ${dataRoot}`);
	console.log(`[smoke] categories: ${categories.length}`);
	console.log(`[smoke] conferences: ${conferences.length}`);
}

main().catch((error) => {
	console.error("[smoke] failed:", error);
	process.exit(1);
});
