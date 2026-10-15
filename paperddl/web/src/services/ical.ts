import type { Conference } from "../types/conference";
import { conferenceKey } from "../utils/conferenceKey";

export type DeadlineMode = "full" | "abstract";

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function toIcsDateTimeUtc(date: Date): string {
  return (
    `${date.getUTCFullYear()}${pad2(date.getUTCMonth() + 1)}${pad2(date.getUTCDate())}` +
    `T${pad2(date.getUTCHours())}${pad2(date.getUTCMinutes())}${pad2(date.getUTCSeconds())}Z`
  );
}

function parseDate(value?: string | null, timezone?: string): Date | null {
  if (!value || value.toUpperCase() === "TBD") return null;
  const parts = value.split(" ");
  if (parts.length === 2 && timezone) {
    const tzOffset = (timezone.toUpperCase() === "AOE") ? -12 : (parseInt(timezone.match(/UTC([+-]\d+)/i)?.[1] || "0", 10));
    const [y, mo, d] = parts[0].split("-").map(Number);
    const [h, mi, s] = parts[1].split(":").map(Number);
    return new Date(Date.UTC(y, mo - 1, d, h - tzOffset, mi, s));
  }
  const parsed = Date.parse(value.replace(" ", "T"));
  return Number.isNaN(parsed) ? null : new Date(parsed);
}

function icsEscape(value: string): string {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/\n/g, "\\n")
    .replace(/;/g, "\\;")
    .replace(/,/g, "\\,");
}

function getDeadline(conference: Conference, mode: DeadlineMode): string | null {
  const anyConf = conference as any;
  const edition = anyConf?.edition || conference.confs?.[0] || null;
  const timeline = edition?.timeline?.[0] || null;
  if (!timeline) return conference.nextDeadline || null;
  if (mode === "abstract") return timeline.abstract_deadline || null;
  return timeline.deadline || conference.nextDeadline || null;
}

export function buildIcs(conferences: Conference[], mode: DeadlineMode): string {
  const now = new Date();
  const dtStamp = toIcsDateTimeUtc(now);
  const lines: string[] = [];

  lines.push("BEGIN:VCALENDAR");
  lines.push("VERSION:2.0");
  lines.push("PRODID:-//paperddl//EN");
  lines.push("CALSCALE:GREGORIAN");
  lines.push("METHOD:PUBLISH");

  for (const conf of conferences) {
    const deadline = getDeadline(conf, mode);
    const timezone = conf.confs?.[0]?.timezone || conf.timezone;
    const start = parseDate(deadline, timezone);
    if (!start) continue;
    const end = new Date(start.getTime() + 60 * 60 * 1000);

    const anyConf = conf as any;
    const year = anyConf?.displayYear || conf.confs?.[0]?.year || "";
    const summary = `${conf.title}${year ? ` ${year}` : ""} (${mode === "abstract" ? "Abstract" : "Full Paper"} Deadline)`;
  const uid = `${conferenceKey(conf)}@paperddl`;

    lines.push("BEGIN:VEVENT");
    lines.push(`UID:${icsEscape(uid)}`);
    lines.push(`DTSTAMP:${dtStamp}`);
    lines.push(`DTSTART:${toIcsDateTimeUtc(start)}`);
    lines.push(`DTEND:${toIcsDateTimeUtc(end)}`);
    lines.push(`SUMMARY:${icsEscape(summary)}`);
    const link = conf.confs?.[0]?.link || "";
    if (link) lines.push(`URL:${icsEscape(link)}`);
    const description = [
      conf.description || "",
      `CCF: ${conf.rank?.ccf || "N"}`,
      `CORE: ${conf.rank?.core || "N"}`,
      `THCPL: ${(conf.rank as any)?.thcpl || (conf.rank as any)?.thc || "N"}`
    ]
      .filter(Boolean)
      .join("\\n");
    if (description) lines.push(`DESCRIPTION:${icsEscape(description)}`);
    lines.push("END:VEVENT");
  }

  lines.push("END:VCALENDAR");
  return lines.join("\r\n") + "\r\n";
}

export function downloadIcs(ics: string, fileName: string): void {
  const blob = new Blob([ics], { type: "text/calendar;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
