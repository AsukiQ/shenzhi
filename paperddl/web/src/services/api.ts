import type { Category, Conference, ConferenceQuery } from "../types/conference";
import type { PaperQuery, PaperSearchResult, PaperTrack, PaperVenue } from "../types/paper.ts";

const API_BASE = (import.meta.env.VITE_API_BASE as string) || "/api";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    credentials: "include"
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchCategories(): Promise<Category[]> {
  return getJson<Category[]>(`${API_BASE}/categories`);
}

export async function fetchConferences(query: ConferenceQuery): Promise<Conference[]> {
  const params = new URLSearchParams();
  if (query.sub) {
    const subValue = Array.isArray(query.sub) ? query.sub.join(",") : query.sub;
    params.set("sub", subValue);
  }
  if (query.q) params.set("q", query.q);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return getJson<Conference[]>(`${API_BASE}/conferences${suffix}`);
}

export async function addConference(conferenceData: any): Promise<any> {
  const res = await fetch(`${API_BASE}/conferences`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(conferenceData),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.message || "Failed to add conference");
  }
  return res.json();

}

export async function fetchPaperVenues(): Promise<PaperVenue[]> {
  return getJson<PaperVenue[]>(`${API_BASE}/papers/venues`);
}

export async function fetchPaperTracks(conference: string): Promise<PaperTrack[]> {
  const params = new URLSearchParams({ conference });
  return getJson<PaperTrack[]>(`${API_BASE}/papers/tracks?${params.toString()}`);
}

export async function fetchPapers(query: PaperQuery): Promise<PaperSearchResult> {
  const params = new URLSearchParams();
  if (query.q) params.set("q", query.q);
  if (query.conference) params.set("conference", query.conference);
  if (query.year) params.set("year", String(query.year));
  if (query.track) params.set("track", query.track);
  if (query.limit) params.set("limit", String(query.limit));
  if (query.offset) params.set("offset", String(query.offset));

  const suffix = params.toString() ? `?${params.toString()}` : "";
  return getJson<PaperSearchResult>(`${API_BASE}/papers${suffix}`);
}
