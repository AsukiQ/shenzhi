import { execFile } from "node:child_process";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

function normalizeArxivIdFromUrl(url) {
  const match = String(url || "").match(/arxiv\.org\/(?:html|abs)\/([^?#/]+)/i);
  return match ? match[1] : "";
}

function safeOutputName(arxivId) {
  return arxivId.replace(/[\\/]/g, "_");
}

export async function runKgEnrich({ arxivHtmlUrl, dryRun = true, maxSections = 5 }) {
  const url = typeof arxivHtmlUrl === "string" ? arxivHtmlUrl.trim() : "";
  if (!url) {
    const error = new Error("arxivHtmlUrl is required");
    error.statusCode = 400;
    throw error;
  }
  if (!/^https:\/\/arxiv\.org\/html\//i.test(url)) {
    const error = new Error("Only https://arxiv.org/html/... URLs are supported");
    error.statusCode = 400;
    throw error;
  }

  const apiKgDir = process.env.API_KG_DIR || "/app/tools/API_KG";
  const pythonBin = process.env.API_KG_PYTHON || "/opt/api_kg_venv/bin/python";
  const arxivId = normalizeArxivIdFromUrl(url);
  const outputPath = path.join(apiKgDir, "data", `${safeOutputName(arxivId)}_kg.json`);
  const sectionLimit = Math.min(Math.max(Number(maxSections) || 5, 1), 12);
  const args = ["main_arxiv_html.py", "--url", url, "--max-sections", String(sectionLimit)];

  if (dryRun !== false) {
    args.push("--dry-run");
  }

  await access(apiKgDir);

  const { stdout, stderr } = await execFileAsync(pythonBin, args, {
    cwd: apiKgDir,
    timeout: 5 * 60 * 1000,
    maxBuffer: 10 * 1024 * 1024,
    env: {
      ...process.env,
      ARXIV_HTML_CACHE_FIRST: process.env.ARXIV_HTML_CACHE_FIRST || "1"
    }
  });

  const raw = await readFile(outputPath, "utf8");
  return {
    ...JSON.parse(raw),
    runner: {
      dryRun: dryRun !== false,
      maxSections: sectionLimit,
      stdout,
      stderr
    }
  };
}
