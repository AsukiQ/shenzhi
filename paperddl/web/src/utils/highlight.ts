export function escapeHtml(input: string): string {
  return input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeRegExp(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function highlightText(text: string, query: string): string {
  const q = query.trim();
  if (!q) return escapeHtml(text);
  const safeText = escapeHtml(text);
  const re = new RegExp(escapeRegExp(escapeHtml(q)), "ig");
  return safeText.replace(re, (match) => `<mark class="search-hit">${match}</mark>`);
}
