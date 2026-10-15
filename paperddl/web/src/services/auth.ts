import { ref } from "vue";

const API_BASE = (import.meta.env.VITE_API_BASE as string) || "/api";

export interface AuthUser {
  id: number;
  username: string;
  role: string;
}

export const currentUser = ref<AuthUser | null>(null);

let userLoaded = false;

async function readJsonError(response: Response, fallback: string) {
  const error = await response.json().catch(() => ({}));
  return error.message || fallback;
}

export async function loadCurrentUser(force = false): Promise<AuthUser | null> {
  if (userLoaded && !force) return currentUser.value;

  const response = await fetch(`${API_BASE}/auth/me`, {
    credentials: "include"
  });

  userLoaded = true;

  if (response.status === 401) {
    currentUser.value = null;
    return null;
  }

  if (!response.ok) {
    currentUser.value = null;
    throw new Error(await readJsonError(response, "Failed to load current user"));
  }

  const data = await response.json() as { user: AuthUser };
  currentUser.value = data.user;
  return data.user;
}

export async function login(username: string, password: string): Promise<AuthUser> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ username, password })
  });

  if (!response.ok) {
    throw new Error(await readJsonError(response, "Failed to login"));
  }

  const data = await response.json() as { user: AuthUser };
  currentUser.value = data.user;
  userLoaded = true;
  return data.user;
}

export async function register(username: string, password: string): Promise<AuthUser> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ username, password })
  });

  if (!response.ok) {
    throw new Error(await readJsonError(response, "Failed to register"));
  }

  const data = await response.json() as { user: AuthUser };
  currentUser.value = data.user;
  userLoaded = true;
  return data.user;
}

export async function logout(): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    credentials: "include"
  });
  currentUser.value = null;
  userLoaded = true;
}
