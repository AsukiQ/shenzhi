import fs from "node:fs/promises";
import path from "node:path";
import yaml from "js-yaml";

const YAML_EXT = ".yml";

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

export async function loadCategories(dataRoot) {
	const typesPath = path.join(dataRoot, "types.yml");
	return readYamlFile(typesPath);
}

export async function loadConferences(dataRoot) {
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

	return { file: path.relative(dataRoot, targetFile) };
}
