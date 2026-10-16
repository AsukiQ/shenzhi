import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import "./env.js";
import rateLimit from "express-rate-limit";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  authenticate,
  authCookieName,
  clearAuthCookie,
  getCookieOptions,
  requireAdmin,
  signAuthToken
} from "./auth.js";
import {
  loadCategories,
  loadConferences,
  withComputedDeadline,
  saveConference
} from "./repository.js";
import {
  findUserByUsername,
  toPublicUser,
  updateLastLogin,
  verifyPassword,
  createUser
} from "./userRepository.js";
import {
  listFavoriteIds,
  addFavorite,
  removeFavorite
} from "./favoriteRepository.js";
import {
  listPapers,
  listPaperTracks,
  listPaperVenues
} from "./paperRepository.js";
import { paperHealthcheck } from "./paperDb.js";
import { registerKgRoutes } from "./kgRoutes.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const serverRoot = path.resolve(__dirname, "..");

const dataRoot = process.env.DATA_ROOT
  ? path.resolve(serverRoot, process.env.DATA_ROOT)
  : path.resolve(serverRoot, "../conference");

const app = express();
const port = Number(process.env.PORT || 5000);

// Requests reach Express through the local Nginx reverse proxy. Trust one
// proxy hop so rate limiting can safely use the client IP from X-Forwarded-For.
app.set("trust proxy", 1);

function getCorsOrigin() {
  const origin = process.env.CORS_ORIGIN;
  if (!origin) return true;
  return origin.split(",").map((item) => item.trim()).filter(Boolean);
}

const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: "Too many attempts, please try again later" }
});

app.use(cors({ origin: getCorsOrigin(), credentials: true }));
app.use(cookieParser());
app.use(express.json());
app.use((req, res, next) => {
  const started = performance.now();
  res.on("finish", () => {
    console.info(`[api-timing] ${req.method} ${req.originalUrl} status=${res.statusCode} duration_ms=${(performance.now() - started).toFixed(1)} remote=${req.ip}`);
  });
  next();
});

registerKgRoutes(app);

app.get("/api/health", (_req, res) => {
  try {
    res.json({ ok: true, time: new Date().toISOString(), papers: paperHealthcheck() });
  } catch (error) {
    res.status(503).json({ ok: false, time: new Date().toISOString(), papers: { status: "error", error: error.message } });
  }
});

app.post("/api/auth/login", loginLimiter, async (req, res) => {
  try {
    const username = typeof req.body?.username === "string" ? req.body.username.trim() : "";
    const password = typeof req.body?.password === "string" ? req.body.password : "";

    if (!username || !password) {
      return res.status(400).json({ message: "Username and password are required" });
    }

    const user = await findUserByUsername(username);
    const passwordMatches = user && Number(user.is_active) === 1
      ? await verifyPassword(user, password)
      : false;

    if (!passwordMatches) {
      return res.status(401).json({ message: "Invalid username or password" });
    }

    const publicUser = toPublicUser(user);
    const token = signAuthToken(publicUser);
    await updateLastLogin(user.id);

    res.cookie(authCookieName, token, getCookieOptions());
    res.json({ user: publicUser });
  } catch (error) {
    res.status(500).json({
      message: "Failed to login",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post("/api/auth/register", loginLimiter, async (req, res) => {
  try {
    const username = typeof req.body?.username === "string" ? req.body.username.trim() : "";
    const password = typeof req.body?.password === "string" ? req.body.password : "";

    if (!username || !password) {
      return res.status(400).json({ message: "用户名和密码是必填项" });
    }

    if (username.length < 3) {
      return res.status(400).json({ message: "用户名长度不能少于3位" });
    }

    if (password.length < 6) {
      return res.status(400).json({ message: "密码长度不能少于6位" });
    }

    const existingUser = await findUserByUsername(username);
    if (existingUser) {
      return res.status(409).json({ message: "用户名已存在" });
    }

    const user = await createUser(username, password);
    const publicUser = toPublicUser(user);
    const token = signAuthToken(publicUser);

    res.cookie(authCookieName, token, getCookieOptions());
    res.status(201).json({ user: publicUser });
  } catch (error) {
    res.status(500).json({
      message: "注册失败",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post("/api/auth/logout", (_req, res) => {
  clearAuthCookie(res);
  res.json({ ok: true });
});

app.get("/api/auth/me", authenticate, (req, res) => {
  res.json({ user: req.user });
});

app.get("/api/favorites", authenticate, async (req, res) => {
  try {
    const favoriteIds = await listFavoriteIds(req.user.id);
    res.json({ favoriteIds });
  } catch (error) {
    res.status(500).json({
      message: "Failed to load favorites",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post("/api/favorites", authenticate, async (req, res) => {
  try {
    const conferenceId =
      typeof req.body?.conferenceId === "string" ? req.body.conferenceId.trim() : "";

    if (!conferenceId) {
      return res.status(400).json({ message: "conferenceId is required" });
    }

    await addFavorite(req.user.id, conferenceId);
    res.status(201).json({ ok: true, conferenceId });
  } catch (error) {
    res.status(500).json({
      message: "Failed to add favorite",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.delete("/api/favorites/:conferenceId", authenticate, async (req, res) => {
  try {
    const conferenceId =
      typeof req.params?.conferenceId === "string" ? req.params.conferenceId.trim() : "";

    if (!conferenceId) {
      return res.status(400).json({ message: "conferenceId is required" });
    }

    await removeFavorite(req.user.id, conferenceId);
    res.json({ ok: true, conferenceId });
  } catch (error) {
    res.status(500).json({
      message: "Failed to remove favorite",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("/api/papers/venues", async (_req, res) => {
  try {
    const recordType = typeof _req.query.record_type === "string" ? _req.query.record_type : "canonical";
    const venues = await listPaperVenues(recordType);
    res.json(venues);
  } catch (error) {
    res.status(500).json({
      message: "Failed to load paper venues",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("/api/papers/tracks", async (req, res) => {
  try {
    const conference = typeof req.query.conference === "string"
      ? req.query.conference.trim()
      : "";

    if (!conference) {
      return res.status(400).json({ message: "conference is required" });
    }

    const recordType = typeof req.query.record_type === "string" ? req.query.record_type : "canonical";
    const tracks = await listPaperTracks(conference, recordType);
    res.json(tracks);
  } catch (error) {
    res.status(500).json({
      message: "Failed to load paper tracks",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("/api/papers", async (req, res) => {
  try {
    const result = await listPapers({
      q: typeof req.query.q === "string" ? req.query.q.trim() : "",
      conference: typeof req.query.conference === "string" ? req.query.conference.trim() : "",
      year: typeof req.query.year === "string" ? req.query.year.trim() : "",
      track: typeof req.query.track === "string" ? req.query.track.trim() : "",
      recordType: typeof req.query.record_type === "string" ? req.query.record_type.trim() : "canonical",
      limit: typeof req.query.limit === "string" ? req.query.limit : "",
      offset: typeof req.query.offset === "string" ? req.query.offset : ""
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({
      message: "Failed to load papers",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("/api/categories", async (_req, res) => {
  try {
    const categories = await loadCategories(dataRoot);
    res.json(categories);
  } catch (error) {
    res.status(500).json({
      message: "Failed to load categories",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("/api/conferences", async (req, res) => {
  try {
    const subParam = req.query.sub;
    const subList = Array.isArray(subParam)
      ? subParam
      : typeof subParam === "string"
        ? subParam.split(",")
        : [];
    const normalizedSubs = subList.map((item) => item.trim()).filter(Boolean);
    const q = typeof req.query.q === "string" ? req.query.q.trim().toLowerCase() : "";
    let conferences = await loadConferences(dataRoot);

    if (normalizedSubs.length) {
      conferences = conferences.filter((item) => normalizedSubs.includes(item.sub));
    }

    if (q) {
      conferences = conferences.filter((item) => {
        const text = `${item.title} ${item.description || ""}`.toLowerCase();
        return text.includes(q);
      });
    }

    conferences = conferences.map(withComputedDeadline);

    res.json(conferences);
  } catch (error) {
    res.status(500).json({
      message: "Failed to load conferences",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post("/api/conferences", authenticate, requireAdmin, async (req, res) => {
  try {
    const result = await saveConference(dataRoot, req.body);
    res.status(201).json({ message: "Conference added successfully", ...result });
  } catch (error) {
    res.status(500).json({
      message: "Failed to add conference",
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

async function startServer() {
  // Prime the read-mostly metadata caches before accepting traffic. The first
  // venue aggregate otherwise has to scan the full SQLite paper table and the
  // first conference request has to parse every YAML file.
  const warmup = await Promise.allSettled([
    loadCategories(dataRoot),
    loadConferences(dataRoot),
    Promise.resolve().then(() => listPaperVenues("canonical"))
  ]);
  const failed = warmup.filter((item) => item.status === "rejected").length;
  if (failed) console.warn(`[paperddl-server] metadata warmup completed with ${failed} failure(s)`);
  else console.log("[paperddl-server] metadata caches warmed");

  app.listen(port, () => {
    console.log(`[paperddl-server] listening on http://localhost:${port}`);
    console.log(`[paperddl-server] data root: ${dataRoot}`);
  });
}

startServer().catch((error) => {
  console.error("[paperddl-server] startup failed", error);
  process.exitCode = 1;
});
