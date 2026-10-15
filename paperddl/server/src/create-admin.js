import { pool } from "./db.js";
import { createAdminUser, findUserByUsername } from "./userRepository.js";

async function main() {
	const username = process.env.ADMIN_USERNAME || process.argv[2];
	const password = process.env.ADMIN_PASSWORD || process.argv[3];

	if (!username || !password) {
		console.error("Usage: npm run auth:create-admin -- <username> <password>");
		console.error("Or set ADMIN_USERNAME and ADMIN_PASSWORD in the environment.");
		process.exitCode = 1;
		return;
	}

	const existing = await findUserByUsername(username);
	if (existing) {
		console.error(`User already exists: ${username}`);
		process.exitCode = 1;
		return;
	}

	const user = await createAdminUser(username, password);
	console.log(`Created admin user: ${user.username} (id: ${user.id})`);
}

main()
	.catch((error) => {
		console.error("Failed to create admin user:", error);
		process.exitCode = 1;
	})
	.finally(async () => {
		await pool.end();
	});
