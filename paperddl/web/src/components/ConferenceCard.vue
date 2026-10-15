<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import type { Conference } from "../types/conference";
import type { DeadlineMode } from "../services/ical";
import { highlightText } from "../utils/highlight";

const props = withDefaults(
  defineProps<{
    conference: Conference;
    highlight?: string;
    deadlineMode?: DeadlineMode;
    isFavorite?: boolean;
    favoritePending?: boolean;
  }>(),
  {
    highlight: "",
    deadlineMode: "full",
    isFavorite: false,
    favoritePending: false
  }
);

const emit = defineEmits<{
  (e: "toggle-favorite", conference: Conference): void;
}>();

const now = ref(Date.now());
let timer: number;

onMounted(() => {
  timer = window.setInterval(() => {
    now.value = Date.now();
  }, 1000);
});

onUnmounted(() => {
  clearInterval(timer);
});

function getTzOffset(tz?: string): number {
  if (!tz) return 0;
  if (tz.toUpperCase() === "AOE") return -12;
  const match = tz.match(/UTC([+-]\d+)/i);
  return match ? parseInt(match[1], 10) : 0;
}

function parseDate(value?: string | null, timezone?: string): Date | null {
  if (!value || value.toUpperCase() === "TBD") return null;
  const tzOffset = getTzOffset(timezone);
  const parts = value.split(" ");
  if (parts.length === 2 && timezone) {
    const [datePart, timePart] = parts;
    const [y, m, d] = datePart.split("-").map(Number);
    const [hh, mm, ss] = timePart.split(":").map(Number);
    return new Date(Date.UTC(y, m - 1, d, hh - tzOffset, mm, ss));
  }
  const parsed = Date.parse(value.replace(" ", "T"));
  return Number.isNaN(parsed) ? null : new Date(parsed);
}

function formatTime(diff: number): string {
  if (diff <= 0) return "00天 00时 00分 00秒";
  const d = Math.floor(diff / (1000 * 60 * 60 * 24));
  const h = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const m = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const s = Math.floor((diff % (1000 * 60)) / 1000);
  return `${String(d).padStart(2, '0')}天 ${String(h).padStart(2, '0')}时 ${String(m).padStart(2, '0')}分 ${String(s).padStart(2, '0')}秒`;
}

function formatDateTime(date: Date, useUtc: boolean): string {
  const year = useUtc ? date.getUTCFullYear() : date.getFullYear();
  const month = String((useUtc ? date.getUTCMonth() : date.getMonth()) + 1).padStart(2, "0");
  const day = String(useUtc ? date.getUTCDate() : date.getDate()).padStart(2, "0");
  const hours = String(useUtc ? date.getUTCHours() : date.getHours()).padStart(2, "0");
  const minutes = String(useUtc ? date.getUTCMinutes() : date.getMinutes()).padStart(2, "0");
  const seconds = String(useUtc ? date.getUTCSeconds() : date.getSeconds()).padStart(2, "0");
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

function formatOffset(date: Date): string {
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const absMinutes = Math.abs(offsetMinutes);
  const hours = String(Math.floor(absMinutes / 60)).padStart(2, "0");
  const minutes = String(absMinutes % 60).padStart(2, "0");
  return minutes === "00" ? `UTC${sign}${Number(hours)}` : `UTC${sign}${hours}:${minutes}`;
}

const isExpired = computed(() => {
  const d = deadlineDate.value;
  if (!d) return false;
  return Boolean(d && d.getTime() <= now.value);
});

const isTbd = computed(() => {
  return !mainDeadline.value || mainDeadline.value.toUpperCase() === "TBD";
});

const isUrgent = computed(() => {
  if (isTbd.value || isExpired.value) return false;
  const d = deadlineDate.value;
  if (!d) return false;
  const diff = d.getTime() - now.value;
  return diff > 0 && diff < 24 * 60 * 60 * 1000;
});

const isWarning = computed(() => {
  if (isTbd.value || isExpired.value) return false;
  const d = deadlineDate.value;
  if (!d) return false;
  const diff = d.getTime() - now.value;
  return diff > 0 && diff <= 7 * 24 * 60 * 60 * 1000;
});

const currentEditionIndex = ref(0);

const currentEdition = computed(() => {
  return props.conference.confs[currentEditionIndex.value] || props.conference.confs[0] || null;
});

const totalEditions = computed(() => {
  return props.conference.confs.length;
});

function prevEdition() {
  if (currentEditionIndex.value < totalEditions.value - 1) {
    currentEditionIndex.value++;
  }
}

function nextEdition() {
  if (currentEditionIndex.value > 0) {
    currentEditionIndex.value--;
  }
}

const timeline = computed(() => {
  return currentEdition.value?.timeline?.[0] || null;
});

const latestYear = computed(() => {
  return currentEdition.value?.year || "";
});

const categoryNameMap: Record<string, string> = {
  DS: "计算机体系结构/并行与分布计算/存储系统",
  NW: "计算机网络",
  SC: "网络与信息安全",
  SE: "软件工程/系统软件/程序设计语言",
  DB: "数据库/数据挖掘/内容检索",
  CT: "计算机科学理论",
  CG: "计算机图形学与多媒体",
  AI: "人工智能",
  HI: "人机交互与普适计算",
  MX: "交叉/综合/新兴"
};

const categoryLabel = computed(() => {
  return categoryNameMap[props.conference.sub] || props.conference.sub || "";
});

const commentText = computed(() => timeline.value?.comment || "");

const ccfRankLabel = computed(() => {
  const rank = props.conference.rank?.ccf || "N";
  if (rank === "A") return "CCF A";
  if (rank === "B") return "CCF B";
  if (rank === "C") return "CCF C";
  return "Non-CCF";
});

const coreRankLabel = computed(() => {
  const core = props.conference.rank?.core || "N";
  return core === "N" ? "Non-CORE" : `CORE ${core}`;
});

const thcplRankLabel = computed(() => {
  const thcpl = (props.conference.rank as any)?.thcpl || (props.conference.rank as any)?.thc || "N";
  return thcpl === "N" ? "Non-THCPL" : `THCPL ${thcpl}`;
});

const mainDeadline = computed(() => {
  const t = timeline.value as any;
  if (props.deadlineMode === "abstract") {
    return t?.abstract_deadline || null;
  }
  return t?.deadline || props.conference.nextDeadline || null;
});

const deadlineLabel = computed(() => {
  return props.deadlineMode === "abstract" ? "摘要截止时间" : "全文截稿时间";
});

const deadlineDate = computed(() => {
  const tz = currentEdition.value?.timezone || props.conference.timezone;
  return parseDate(mainDeadline.value || undefined, tz);
});

const deadlineCountdown = computed(() => {
  if (!deadlineDate.value) return "TBD";
  return formatTime(deadlineDate.value.getTime() - now.value);
});

const deadlineLine = computed(() => {
  if (!deadlineDate.value) return "TBD";
  const local = formatDateTime(deadlineDate.value, false);
  const localOffset = formatOffset(deadlineDate.value);
  const origTime = mainDeadline.value || "";
  const origTz = currentEdition.value?.timezone || props.conference.timezone || "Local";
  return `${local} ${localOffset} (${origTime} ${origTz})`;
});

const countdownClass = computed(() => {
  if (!deadlineDate.value || isExpired.value || isTbd.value) return "countdown-normal";
  const diff = deadlineDate.value.getTime() - now.value;
  if (diff <= 24 * 60 * 60 * 1000) return "countdown-urgent";
  if (diff <= 7 * 24 * 60 * 60 * 1000) return "countdown-warning";
  if (diff <= 30 * 24 * 60 * 60 * 1000) return "countdown-attention";
  return "countdown-normal";
});

const timelinePoints = computed(() => {
  const points: { label: string; time: Date; type: "abstract" | "deadline" }[] = [];
  const edition = currentEdition.value as any;
  if (!edition?.timeline) return [];
  edition.timeline.forEach((item: any) => {
    const abs = parseDate(item.abstract_deadline);
    const ddl = parseDate(item.deadline);
    if (abs) {
      points.push({ label: `注册截止: ${formatDateTime(abs, false)}`, time: abs, type: "abstract" });
    }
    if (ddl) {
      points.push({ label: `投稿截止: ${formatDateTime(ddl, false)}`, time: ddl, type: "deadline" });
    }
  });
  return points.sort((a, b) => a.time.getTime() - b.time.getTime());
});

const timelineRange = computed(() => {
  if (timelinePoints.value.length === 0) return null;
  const first = timelinePoints.value[0].time.getTime();
  const last = timelinePoints.value[timelinePoints.value.length - 1].time.getTime();
  const sevenDays = 7 * 24 * 60 * 60 * 1000;
  return {
    start: first - sevenDays,
    end: last + sevenDays,
    last
  };
});

const timelineStyles = computed(() => {
  const range = timelineRange.value;
  if (!range) return { canLine: "width:0%;", nowPercent: 0 };
  const nowMs = now.value;
  const total = range.end - range.start;
  const left = Math.min(100, Math.max(0, ((nowMs - range.start) / total) * 100));
  const width = Math.max(0, Math.min(100 - left, ((range.last - nowMs) / total) * 100));
  return { canLine: `width:${width}%;left:${left}%;max-width:${100 - left}%;`, nowPercent: left };
});

const timelineMarkers = computed(() => {
  const range = timelineRange.value;
  if (!range) return [];
  const total = range.end - range.start;
  const raw = timelinePoints.value.map((point) => {
    const percent = Math.min(99.5, Math.max(0.5, ((point.time.getTime() - range.start) / total) * 100));
    const month = String(point.time.getMonth() + 1).padStart(2, "0");
    const day = String(point.time.getDate()).padStart(2, "0");
    return {
      percent,
      dateText: `${month}/${day}`,
      label: point.label,
      type: point.type,
      showLabel: true
    };
  });

  // If markers are too close, keep only the later marker's label to avoid overlap.
  const minGapPercent = 5;
  for (let i = 1; i < raw.length; i++) {
    const prev = raw[i - 1];
    const curr = raw[i];
    if (curr.percent - prev.percent < minGapPercent) {
      prev.showLabel = false;
      curr.showLabel = true;
    }
  }

  return raw;
});

const titleHtml = computed(() => {
  const anyConf = props.conference as any;
  const year = anyConf?.displayYear || latestYear.value || "";
  const title = `${props.conference.title}${year ? ` ${year}` : ""}`;
  return highlightText(title, props.highlight || "");
});

const descriptionHtml = computed(() => {
  return highlightText(props.conference.description || "", props.highlight || "");
});

function handleFavoriteClick() {
  emit("toggle-favorite", props.conference);
}
</script>

<template>
  <article
    class="conf-card"
    :class="{ 'conf-urgent': isUrgent, 'conf-warning': isWarning && !isUrgent, 'conf-expired': isExpired }"
  >
    <span class="status-bar" aria-hidden="true"></span>
    <div class="conf-cell">
      <div class="thaw-table-cell-layout">
        <div :class="['conf-content', { 'conf-fin': isExpired }]">
          <div class="conf-title-row"><div class="conf-title">
            <a
              v-if="currentEdition?.link"
              :href="currentEdition.link"
              target="_blank"
            >
              <span v-html="titleHtml"></span>
            </a>
            <span v-else v-html="titleHtml"></span>
            <span v-if="totalEditions > 1" class="edition-nav">
              <button
                class="nav-btn"
                :disabled="currentEditionIndex === totalEditions - 1"
                @click="prevEdition"
                title="上一年">
                &larr;
              </button>
              <button
                class="nav-btn"
                :disabled="currentEditionIndex === 0"
                @click="nextEdition"
                title="下一年">
                &rarr;
              </button>
            </span>
          </div>
            <button
              class="favorite-btn"
              :class="{ 'favorite-btn--active': isFavorite }"
              type="button"
              :disabled="favoritePending"
              :aria-pressed="isFavorite"
              :title="isFavorite ? '取消收藏' : '收藏会议'"
              @click.stop="handleFavoriteClick"
            >
              <span aria-hidden="true">{{ isFavorite ? "★" : "☆" }}</span>
              <span class="favorite-btn__text">{{ isFavorite ? "已收藏" : "收藏" }}</span>
            </button>
          </div>

          <div class="conf-meta-line" v-if="currentEdition?.date || currentEdition?.place">
            {{ currentEdition?.date }} {{ currentEdition?.place }}
          </div>

          <div class="conf-desc" v-if="conference.description">
            <span v-html="descriptionHtml"></span>
          </div>

          <div class="tag-container">
            <span class="plain-tag">{{ ccfRankLabel }}</span>
            <span class="plain-tag">{{ coreRankLabel }}</span>
            <span class="plain-tag">{{ thcplRankLabel }}</span>
            <span v-if="commentText" class="note-text"><b>NOTE: </b>{{ commentText }}</span>
          </div>

          <div class="conf-subname-line">
            <span v-if="(conference as any).acc_str">Acc. Rate: {{ (conference as any).acc_str }}</span>
            <span class="subname-badge">{{ categoryLabel }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="conf-cell">
      <div class="thaw-table-cell-layout">
        <div :class="['conf-content', { 'conf-fin': isExpired }]">
          <div class="countdown-container">
            <div class="countdown-display">
              <span class="countdown-value" :class="countdownClass">{{ deadlineCountdown }}</span>
            </div>
          </div>

          <div class="conf-meta-line">
            <span v-if="isTbd">
              {{ deadlineLabel }}:
              <span>待定</span>
            </span>
            <span v-else>{{ deadlineLabel }}: {{ deadlineLine }}</span>
          </div>

          <div class="conf-meta-line" v-if="conference.confs?.[0]?.link">
            官网:
            <a :href="conference.confs[0].link" target="_blank">{{ conference.confs[0].link }}</a>
          </div>

          <div v-if="!isExpired && !isTbd && timelinePoints.length" class="time_con">
            <div class="line_time">
              <div class="all_line">
                <div class="line">
                  <div class="can_line" :style="timelineStyles.canLine"></div>
                  <div
                    v-for="(marker, index) in timelineMarkers"
                    :key="`marker-${index}`"
                    :class="marker.type === 'abstract' ? 'square square_all' : 'dot dot_all'"
                    :style="{ left: `${marker.percent}%` }"
                    :title="marker.label"
                  >
                    <em v-if="marker.showLabel" class="marker-date">{{ marker.dateText }}</em>
                  </div>
                  <div class="dot sel_dot" :style="{ left: `${timelineStyles.nowPercent}%` }">
                    <em>当前时间: {{ formatDateTime(new Date(now), false) }}</em>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </article>
</template>

<style scoped>
.search-hit {
  background: rgba(250, 204, 21, 0.35);
  padding: 0 2px;
  border-radius: 2px;
}

.conf-card {
  display: grid;
  grid-template-columns: 1fr 1fr;
  border-bottom: 1px solid #ebeef5;
  position: relative;
  padding-left: 6px;
}

.status-bar {
  position: absolute;
  left: 0;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: 2px;
  background: #67c23a;
}

.conf-warning .status-bar {
  background: #e6a23c;
}

.conf-urgent .status-bar {
  background: #f56c6c;
}

.conf-expired .status-bar {
  background: #909399;
}

.conf-cell {
  padding: 0 12px;
}

.thaw-table-cell-layout {
  display: block;
  padding: 12px 0;
  transition: background-color 150ms ease;
}

.thaw-table-cell-layout:hover {
  background-color: #f5f7fa;
}

.conf-content.conf-fin {
  opacity: 0.4;
}

.conf-title {
  font-size: 20px;
  font-weight: 500;
  color: #2c3e50;
  letter-spacing: 0.2px;
}

.conf-title-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.conf-title a {
  text-decoration: none;
  color: var(--color-primary);
}

.conf-title a:hover {
  text-decoration: underline;
}

.favorite-btn {
  flex: 0 0 auto;
  border: 1px solid rgba(148, 163, 184, 0.45);
  border-radius: 999px;
  background: #ffffff;
  color: #64748b;
  padding: 4px 10px;
  font-size: 13px;
  line-height: 1.4;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  transition: all 0.18s ease;
}

.favorite-btn:hover:not(:disabled) {
  border-color: #f59e0b;
  color: #b45309;
  box-shadow: 0 6px 16px rgba(245, 158, 11, 0.15);
}

.favorite-btn--active {
  border-color: #f59e0b;
  background: #fffbeb;
  color: #b45309;
}

.favorite-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.favorite-btn__text {
  white-space: nowrap;
}

.edition-nav {
  display: inline-flex;
  gap: 4px;
  margin-left: 12px;
  vertical-align: middle;
}

.nav-btn {
  background: none;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  padding: 2px 8px;
  font-size: 0.9em;
  color: var(--text-color);
  transition: all 0.2s;
}

.nav-btn:hover:not(:disabled) {
  background-color: var(--bg-hover);
  color: var(--color-primary);
  border-color: var(--color-primary);
}

.nav-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.conf-meta-line a {
  color: #409eff;
  text-decoration: none;
}

.conf-meta-line a:hover {
  text-decoration: underline;
}

.conf-desc {
  font-size: 14px;
  color: #606266;
  margin-top: 3px;
}

.tag-container {
  margin-top: 3px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}

.plain-tag {
  background-color: #ffffff;
  border: 1px solid #b3d8ff;
  border-radius: 4px;
  height: 20px;
  line-height: 18px;
  padding: 0 5px;
  font-size: 12px;
  color: #409eff;
}

.note-text {
  color: #409eff;
  font-size: 12px;
}

.conf-subname-line {
  padding-top: 5px;
  font-size: 14px;
  color: #606266;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}

.subname-badge {
  color: rgb(36, 101, 191);
  background: rgba(236, 240, 241, 0.7);
  font-size: 13px;
  padding: 3px 5px;
}

.countdown-container {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-start;
}

.countdown-display {
  font-size: 20px;
  font-weight: 500;
  color: #2c3e50;
}

.countdown-value {
  display: inline-flex;
  align-items: center;
  letter-spacing: 0.5px;
}

.countdown-normal {
  color: #666666;
}

.countdown-attention {
  color: #67c23a;
}

.countdown-warning {
  color: #e6a23c;
}

.countdown-urgent {
  color: #f56c6c;
}

.conf-urgent .countdown-display {
  color: #f56c6c;
}

.conf-expired .countdown-display {
  color: #909399;
}

.time_con {
  margin-top: 6px;
}

.line_time {
  position: relative;
  user-select: none;
}

.line_time .all_line {
  width: 90%;
  margin: 0 5%;
  padding-top: 25px;
  padding-bottom: 15px;
}

.line_time .line {
  width: 100%;
  height: 3px;
  background: #ccc;
  position: relative;
}

.line_time .can_line {
  background: #1890ff77;
  height: 3px;
  width: 0%;
  position: absolute;
  left: 0;
}

.line_time .reference {
  width: 1px;
  height: 8px;
  border: 0;
  background: #bbb;
  position: absolute;
  top: -3px;
  white-space: nowrap;
}

.line_time .reference em {
  color: #bbb;
  position: absolute;
  transform: translateX(-50%);
  margin-top: 5px;
  font-size: 12px;
}

.line_time .dot,
.line_time .square {
  width: 8px;
  height: 8px;
  border: 2px solid #4a9eff;
  background: white;
  position: absolute;
  top: -3px;
  margin-left: -4px;
  box-sizing: border-box;
  white-space: nowrap;
}

.line_time .dot {
  border-radius: 50%;
}

.line_time .dot_all em,
.line_time .square_all em {
  display: none;
  color: #409eff;
  transform: translateX(-50%);
  position: absolute;
  top: -25px;
}

.line_time .dot_all em.marker-date,
.line_time .square_all em.marker-date {
  display: inline-block;
  color: #64748b;
  font-size: 11px;
  top: 12px;
  white-space: nowrap;
}

.line_time .dot_all:hover,
.line_time .square_all:hover {
  width: 10px;
  height: 10px;
  border: 2px solid #409eff;
  top: -4px;
}

.line_time .dot_all:hover em,
.line_time .square_all:hover em {
  display: inline-block;
}

.line_time .sel_dot {
  width: 10px;
  height: 10px;
  top: -4px;
  border: 2px solid #ffa500;
  box-shadow: 0 0 10px 4px rgba(255, 163, 2, 0.3);
  z-index: 5;
  position: absolute;
}

.line_time .sel_dot em {
  display: none;
  color: #ffa500;
  transform: translateX(-50%);
  position: absolute;
  top: -25px;
}

.line_time .sel_dot:hover em {
  display: inline-block;
}

@media (max-width: 900px) {
  .conf-card {
    grid-template-columns: 1fr;
  }
  .conf-cell {
    padding: 0 6px;
  }
}
</style>
