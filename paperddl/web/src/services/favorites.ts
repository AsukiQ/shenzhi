const API_BASE = (import.meta.env.VITE_API_BASE as string) || "/api";

async function readJsonError(response: Response, fallback: string) {
  const error = await response.json().catch(() => ({}));
  return error.message || fallback;
}

export async function fetchFavoriteIds(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/favorites`, {
    credentials: "include"
  });

  if (response.status === 401) return [];

  if (!response.ok) {
    throw new Error(await readJsonError(response, "Failed to load favorites"));
  }

  const data = await response.json() as { favoriteIds: string[] };
  return data.favoriteIds || [];
}

export async function addFavorite(conferenceId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/favorites`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ conferenceId })
  });

  if (!response.ok) {
    throw new Error(await readJsonError(response, "Failed to add favorite"));
  }
}

export async function removeFavorite(conferenceId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/favorites/${encodeURIComponent(conferenceId)}`, {
    method: "DELETE",
    credentials: "include"
  });

  if (!response.ok) {
    throw new Error(await readJsonError(response, "Failed to remove favorite"));
  }
}
