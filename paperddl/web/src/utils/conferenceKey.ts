import type { Conference } from "../types/conference";

export function conferenceKey(conference: Conference): string {
  const anyConf = conference as any;
  const editionId = anyConf?.edition?.id || conference.confs?.[0]?.id;
  const year = anyConf?.displayYear || conference.confs?.[0]?.year;
  return `${conference.title}::${editionId || year || ""}`;
}
