import { execute } from "./db.js";

export async function listFavoriteIds(userId) {
  const rows = await execute(
    `SELECT conference_id
     FROM favorites
     WHERE user_id = ?
     ORDER BY created_at DESC`,
    [userId]
  );
  return rows.map((row) => row.conference_id);
}

export async function addFavorite(userId, conferenceId) {
  await execute(
    `INSERT INTO favorites (user_id, conference_id)
     VALUES (?, ?)
     ON DUPLICATE KEY UPDATE created_at = created_at`,
    [userId, conferenceId]
  );
}

export async function removeFavorite(userId, conferenceId) {
  const result = await execute(
    `DELETE FROM favorites
     WHERE user_id = ? AND conference_id = ?`,
    [userId, conferenceId]
  );
  return result.affectedRows || 0;
}
