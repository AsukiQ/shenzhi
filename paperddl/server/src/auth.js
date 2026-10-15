import "./env.js";
import jwt from "jsonwebtoken";
import { findActiveUserById, toPublicUser } from "./userRepository.js";

export const authCookieName = process.env.COOKIE_NAME || "paperddl_token";
const jwtExpiresIn = process.env.JWT_EXPIRES_IN || "7d";

function getJwtSecret() {
	const secret = process.env.JWT_SECRET;
	if (!secret) {
		throw new Error("JWT_SECRET is not configured");
	}
	return secret;
}

function durationToMs(value) {
	if (/^\d+$/.test(value)) return Number(value) * 1000;
	const match = /^(\d+)([smhd])$/.exec(value);
	if (!match) return 7 * 24 * 60 * 60 * 1000;
	const amount = Number(match[1]);
	const unit = match[2];
	if (unit === "s") return amount * 1000;
	if (unit === "m") return amount * 60 * 1000;
	if (unit === "h") return amount * 60 * 60 * 1000;
	return amount * 24 * 60 * 60 * 1000;
}

export function getCookieOptions() {
	return {
		httpOnly: true,
		sameSite: "lax",
		secure: process.env.COOKIE_SECURE === "true",
		maxAge: durationToMs(jwtExpiresIn),
		path: "/"
	};
}

export function signAuthToken(user) {
	return jwt.sign(
		{
			sub: String(user.id),
			username: user.username,
			role: user.role
		},
		getJwtSecret(),
		{ expiresIn: jwtExpiresIn }
	);
}

export function clearAuthCookie(res) {
	res.clearCookie(authCookieName, {
		httpOnly: true,
		sameSite: "lax",
		secure: process.env.COOKIE_SECURE === "true",
		path: "/"
	});
}

export async function authenticate(req, res, next) {
	try {
		const token = req.cookies?.[authCookieName];
		if (!token) {
			return res.status(401).json({ message: "Authentication required" });
		}

		const payload = jwt.verify(token, getJwtSecret());
		const userId = Number(payload.sub);
		if (!Number.isFinite(userId)) {
			clearAuthCookie(res);
			return res.status(401).json({ message: "Invalid session" });
		}

		const user = await findActiveUserById(userId);
		if (!user) {
			clearAuthCookie(res);
			return res.status(401).json({ message: "Invalid session" });
		}

		req.user = toPublicUser(user);
		next();
	} catch (_error) {
		clearAuthCookie(res);
		return res.status(401).json({ message: "Invalid session" });
	}
}

export function requireAdmin(req, res, next) {
	if (req.user?.role !== "admin") {
		return res.status(403).json({ message: "Admin access required" });
	}
	next();
}
