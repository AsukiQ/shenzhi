import bcrypt from "bcryptjs";
import { execute } from "./db.js";

const bcryptRounds = Number(process.env.BCRYPT_ROUNDS || 12);

export function toPublicUser(user) {
	return {
		id: Number(user.id),
		username: user.username,
		role: user.role
	};
}

export async function findUserByUsername(username) {
	const rows = await execute(
		`SELECT id, username, password_hash, role, is_active
		 FROM users
		 WHERE username = ?
		 LIMIT 1`,
		[username]
	);
	return rows[0] || null;
}

export async function findActiveUserById(id) {
	const rows = await execute(
		`SELECT id, username, role, is_active
		 FROM users
		 WHERE id = ? AND is_active = 1
		 LIMIT 1`,
		[id]
	);
	return rows[0] || null;
}

export async function verifyPassword(user, password) {
	return bcrypt.compare(password, user.password_hash);
}

export async function updateLastLogin(userId) {
	await execute("UPDATE users SET last_login_at = NOW() WHERE id = ?", [userId]);
}

export async function createUser(username, password) {
	const passwordHash = await bcrypt.hash(password, bcryptRounds);
	const result = await execute(
		`INSERT INTO users (username, password_hash, role, is_active)
		 VALUES (?, ?, 'user', 1)`,
		[username, passwordHash]
	);
	return {
		id: Number(result.insertId),
		username,
		role: "user"
	};
}

export async function createAdminUser(username, password) {
	const passwordHash = await bcrypt.hash(password, bcryptRounds);
	const result = await execute(
		`INSERT INTO users (username, password_hash, role, is_active)
		 VALUES (?, ?, 'admin', 1)`,
		[username, passwordHash]
	);
	return {
		id: Number(result.insertId),
		username,
		role: "admin"
	};
}
