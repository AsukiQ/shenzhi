import fs from "node:fs/promises";
import path from "node:path";
import yaml from "js-yaml";

const YAML_EXT = ".yml";

// Conference metadata is read-only for normal requests and spans a few
// hundred YAML files. Keep one short lived, shared cache per data root so a
// page refresh does not reopen every file. Writes invalidate both entries.
const metadataCache = new Map();
const CATEGORY_CACHE_TTL_MS = 5 * 60 * 1000;
const CONFERENCE_CACHE_TTL_MS = 60 * 1000;

async function cachedMetadata(key, ttlMs, loader) {
	const now = Date.now();
	const cached = metadataCache.get(key);
	if (cached?.value && now - cached.createdAt < ttlMs) return cached.value;
	if (cached?.promise) return cached.promise;

	const promise = Promise.resolve()
		.then(loader)
		.then((value) => {
			metadataCache.set(key, { createdAt: Date.now(), value });
			return value;
		})
		.catch((error) => {
			metadataCache.delete(key);
			throw error;
		});
	metadataCache.set(key, { createdAt: now, promise });
	return promise;
}

export function clearMetadataCache(dataRoot) {
	const root = path.resolve(dataRoot);
	metadataCache.delete(`categories:${root}`);
	metadataCache.delete(`conferences:${root}`);
}

async function readYamlFile(filePath) {
	const raw = await fs.readFile(filePath, "utf-8");
	const parsed = yaml.load(raw);
	if (!parsed) return [];
	return Array.isArray(parsed) ? parsed : [parsed];
}

async function writeYamlFile(filePath, data) {
	const dump = yaml.dump(data, { lineWidth: 120, noCompatMode: true });
	await fs.writeFile(filePath, dump, "utf-8");
}

function pickDeadline(conf) {
	if (conf.nextDeadline) return conf.nextDeadline;
	const editions = Array.isArray(conf.confs) ? conf.confs : [];
	if (editions.length === 0) return undefined;
	const sorted = editions
		.slice()
		.sort((a, b) => Number(b.year ?? 0) - Number(a.year ?? 0));
	for (const edition of sorted) {
		const timeline = Array.isArray(edition.timeline) ? edition.timeline : [];
		for (const entry of timeline) {
			const value = entry?.deadline || entry?.abstract_deadline;
			if (value && String(value).toUpperCase() !== "TBD") {
				return value;
			}
		}
	}
	return undefined;
}

async function readCategories(dataRoot) {
	const typesPath = path.join(dataRoot, "types.yml");
	return readYamlFile(typesPath);
}

async function readConferences(dataRoot) {
	const entries = await fs.readdir(dataRoot, { withFileTypes: true });
	const conferences = [];

	for (const entry of entries) {
		if (!entry.isDirectory()) continue;
		const sub = entry.name;
		const subDir = path.join(dataRoot, sub);
		const files = await fs.readdir(subDir);
		for (const file of files) {
			if (!file.endsWith(YAML_EXT)) continue;
			const filePath = path.join(subDir, file);
			const items = await readYamlFile(filePath);
			for (const item of items) {
				conferences.push({ ...item, sub: item.sub || sub });
			}
		}
	}

	return conferences;
}

export async function loadCategories(dataRoot) {
	const root = path.resolve(dataRoot);
	return cachedMetadata(`categories:${root}`, CATEGORY_CACHE_TTL_MS, () => readCategories(root));
}

export async function loadConferences(dataRoot) {
	const root = path.resolve(dataRoot);
	return cachedMetadata(`conferences:${root}`, CONFERENCE_CACHE_TTL_MS, () => readConferences(root));
}

export function withComputedDeadline(conf) {
	const nextDeadline = pickDeadline(conf);
	return nextDeadline ? { ...conf, nextDeadline } : { ...conf };
}

export async function saveConference(dataRoot, payload) {
	if (!payload || typeof payload !== "object") {
		throw new Error("Invalid payload");
	}
	const sub = payload.sub;
	if (!sub) {
		throw new Error("Missing sub category");
	}

	const subDir = path.join(dataRoot, sub);
	await fs.mkdir(subDir, { recursive: true });

	const targetFile = path.join(subDir, "manual.yml");
	let list = [];
	try {
		list = await readYamlFile(targetFile);
	} catch (error) {
		if (error.code !== "ENOENT") throw error;
	}

	const normalized = { ...payload, sub };
	list.push(normalized);
	await writeYamlFile(targetFile, list);
	clearMetadataCache(dataRoot);

	return { file: path.relative(dataRoot, targetFile) };
}
