<template>
  <div class="paper-cool">
    <header class="pc-top-nav">
      <div class="pc-top-nav__inner">
        <div class="pc-brand">
          <h2>深知论文</h2>
          <p>会议、年份、track 与论文搜索均来自后端数据库。</p>
        </div>
        <button class="ghost-btn" type="button" @click="$router.push('/')">返回首页</button>
      </div>
    </header>

    <section class="summary-panel">
      <div class="stats-strip">
        <div class="stat-cell">
          <span>论文</span>
          <strong>{{ totalPapers.toLocaleString() }}</strong>
        </div>
        <div class="stat-cell">
          <span>会议</span>
          <strong>{{ venueList.length }}</strong>
        </div>
        <div class="stat-cell">
          <span>年份</span>
          <strong>{{ yearRange }}</strong>
        </div>
        <div class="stat-cell">
          <span>Tracks</span>
          <strong>{{ totalTracks.toLocaleString() }}</strong>
        </div>
      </div>

      <div class="search-panel">
        <div class="pc-search">
          <div class="pc-search__main">
            <input
              v-model="keyword"
              class="pc-search__input"
              placeholder="搜索标题、摘要、作者、关键词或会议"
              @input="schedulePreviewSearch"
              @keyup.enter="openFullSearch"
            />
            <button type="button" class="pc-search__submit" @click="openFullSearch">完整搜索</button>
          </div>

          <div class="pc-search__chips">
            <button v-for="word in quickSearchWords" :key="word" type="button" @click="useQuickSearch(word)">
              {{ word }}
            </button>
          </div>

          <div v-if="keyword.trim()" class="pc-search__preview">
            <div class="preview-head">
              <span v-if="searchLoading">正在查询后端...</span>
              <span v-else>匹配 {{ searchTotal.toLocaleString() }} 篇论文</span>
              <button v-if="searchResults.length" type="button" @click="openFullSearch">查看全部</button>
            </div>
            <div v-if="searchError" class="hint error">{{ searchError }}</div>
            <div v-else-if="!searchLoading && searchResults.length === 0" class="hint">没有匹配结果。</div>
            <div v-else class="preview-list">
	              <article v-for="paper in searchResults" :key="paper.id" class="preview-item">
	                <button type="button" @click="openPaperContext(paper)">
	                  <strong>{{ paper.title }}</strong>
	                  <span>{{ paper.conference }} {{ paper.year }} · {{ paper.track || 'Unknown track' }}</span>
	                  <small>{{ paper.authors.slice(0, 4).join(', ') || 'Unknown authors' }}</small>
	                </button>
	              </article>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="venue-section">
      <div class="section-head">
        <div>
          <h3>会议论文库</h3>
          <p>点击会议查看后端统计出的年份和 track。</p>
        </div>
        <input v-model="venueFilter" placeholder="筛选会议名称或描述" />
      </div>

      <p v-if="loading" class="hint">正在加载会议与论文统计...</p>
      <p v-else-if="error" class="hint error">{{ error }}</p>
      <p v-else-if="filteredVenues.length === 0" class="hint">没有匹配的会议。</p>

      <div class="venue-grid">
        <button
          v-for="venue in filteredVenues"
          :key="venue.conference"
          type="button"
          class="venue-tile"
          @click="goTrack(venue.conference)"
        >
          <span class="venue-name">{{ venue.conference }}</span>
          <span class="venue-desc">{{ venue.meta?.description || '暂无会议描述' }}</span>
          <span class="venue-stat-line">
            {{ venue.count.toLocaleString() }} papers · {{ venue.yearCount }} years · {{ venue.trackCount }} tracks
          </span>
          <span class="venue-stat-line muted">{{ venue.minYear }}-{{ venue.maxYear }}</span>
          <span v-if="venue.meta?.rank" class="rank-line">
            <b>CCF {{ venue.meta.rank.ccf || 'N' }}</b>
            <b v-if="venue.meta.rank.core">CORE {{ venue.meta.rank.core }}</b>
            <b v-if="venue.meta.rank.thcpl || venue.meta.rank.thc">THCPL {{ venue.meta.rank.thcpl || venue.meta.rank.thc }}</b>
          </span>
        </button>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { fetchConferences, fetchPapers, fetchPaperVenues } from "../../services/api";
import type { Conference } from "../../types/conference";
import type { PaperRecord, PaperVenue } from "../../types/paper";

interface EnrichedVenue extends PaperVenue {
  meta?: Conference;
}

const router = useRouter();
const keyword = ref("");
const venueFilter = ref("");
const loading = ref(false);
const error = ref("");
const venueList = ref<EnrichedVenue[]>([]);
const searchResults = ref<PaperRecord[]>([]);
const searchTotal = ref(0);
const searchLoading = ref(false);
const searchError = ref("");
let previewTimer: number | undefined;
let searchSeq = 0;

const quickSearchWords = [
  "transformer",
  "diffusion",
  "large language model",
  "graph neural network",
  "speech recognition",
  "security",
  "optimization",
  "medical image"
];

const totalPapers = computed(() => venueList.value.reduce((sum, venue) => sum + venue.count, 0));
const totalTracks = computed(() => venueList.value.reduce((sum, venue) => sum + venue.trackCount, 0));

const yearRange = computed(() => {
  const years = venueList.value.flatMap((venue) => [venue.minYear, venue.maxYear]).filter(Boolean);
  if (!years.length) return "-";
  return `${Math.min(...years)}-${Math.max(...years)}`;
});

const filteredVenues = computed(() => {
  const q = venueFilter.value.trim().toLowerCase();
  if (!q) return venueList.value;
  return venueList.value.filter((venue) => {
    const description = venue.meta?.description || "";
    return `${venue.conference} ${description}`.toLowerCase().includes(q);
  });
});

function normalizeKey(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, "-");
}

function buildConferenceMap(conferences: Conference[]) {
  const map = new Map<string, Conference>();
  for (const conf of conferences) {
    const title = conf.title || "";
    const firstToken = title.split(/\s+/)[0] || "";
    for (const rawKey of [title, firstToken]) {
      if (!rawKey) continue;
      const key = normalizeKey(rawKey);
      if (!map.has(key)) map.set(key, conf);
    }
  }
  return map;
}

async function loadVenues() {
  loading.value = true;
  error.value = "";
  try {
    const [paperVenues, conferences] = await Promise.all([
      fetchPaperVenues(),
      fetchConferences({ sub: [], q: "" })
    ]);
    const conferenceMap = buildConferenceMap(conferences);
    venueList.value = paperVenues.map((venue) => ({
      ...venue,
      meta: conferenceMap.get(normalizeKey(venue.conference))
    }));
  } catch (err) {
    error.value = err instanceof Error ? err.message : "会议与论文统计加载失败";
    venueList.value = [];
  } finally {
    loading.value = false;
  }
}

async function runPreviewSearch() {
  const q = keyword.value.trim();
  const seq = ++searchSeq;
  searchError.value = "";
  searchTotal.value = 0;

  if (!q) {
    searchResults.value = [];
    searchLoading.value = false;
    return;
  }

  searchLoading.value = true;
  try {
    const result = await fetchPapers({ q, limit: 6, offset: 0 });
    if (seq !== searchSeq) return;
    searchResults.value = result.items;
    searchTotal.value = result.total;
  } catch (err) {
    if (seq !== searchSeq) return;
    searchResults.value = [];
    searchError.value = err instanceof Error ? err.message : "搜索失败";
  } finally {
    if (seq === searchSeq) searchLoading.value = false;
  }
}

function schedulePreviewSearch() {
  window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(runPreviewSearch, 300);
}

function useQuickSearch(word: string) {
  keyword.value = word;
  runPreviewSearch();
}

function openFullSearch() {
  const search = keyword.value.trim();
  if (!search) return;
  router.push({
    name: "detail",
    params: { venue: "global", year: "global", track: "global" },
    query: { search }
  });
}

function openPaperContext(paper: PaperRecord) {
  router.push({
    name: "detail",
    params: {
      venue: paper.conference,
      year: String(paper.year),
      track: paper.track || "all"
    },
    query: { search: keyword.value.trim() || paper.title }
  });
}

function goTrack(name: string) {
  router.push({ name: "track", params: { venue: name } });
}

onMounted(loadVenues);

onBeforeUnmount(() => {
  window.clearTimeout(previewTimer);
});
</script>

<style scoped>
.paper-cool {
  width: min(1440px, calc(100% - 40px));
  margin: 0 auto;
  padding: 24px 20px 40px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.pc-top-nav {
  position: sticky;
  top: 0;
  z-index: 15;
  width: 100vw;
  margin: -24px calc(50% - 50vw) 16px;
  padding: 0 max(20px, calc((100vw - 1440px) / 2 + 20px));
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid rgba(219, 227, 239, 0.9);
  backdrop-filter: blur(18px);
}
.pc-top-nav__inner {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.pc-brand h2 {
  margin: 0;
  font-size: 1.5rem;
  line-height: 1.1;
  letter-spacing: -0.02em;
}
.pc-brand p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
}

.summary-panel {
  display: flex;
  flex-direction: column;
  margin-bottom: 14px;
}
.summary-panel > * + * {
  margin-top: 14px;
}

.summary-panel .stats-strip,
.summary-panel .search-panel {
  margin-bottom: 0;
}
.summary-panel .search-panel {
  padding-left: 10px;
  padding-right: 10px;
}
.ghost-btn {
  border: 1px solid #dbe3ef;
  background: #fff;
  color: #334155;
  border-radius: 999px;
  padding: 10px 18px;
  cursor: pointer;
}
.ghost-btn:hover {
  border-color: #bfd5ff;
  color: #2563eb;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.stats-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  overflow: hidden;
  margin-bottom: 14px;
}
.stat-cell {
  padding: 14px 16px;
  border-right: 1px solid #e2e8f0;
}
.stat-cell:last-child {
  border-right: 0;
}
.stat-cell span {
  display: block;
  color: #64748b;
  font-size: 12px;
  margin-bottom: 4px;
}
.stat-cell strong {
  color: #0f172a;
  font-size: 22px;
}
.search-panel,
.venue-section {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  padding: 16px;
  margin-bottom: 14px;
}
.pc-search {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.pc-search__main {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 128px;
  gap: 12px;
  align-items: center;
}
.pc-search__input,
.section-head input {
  width: 100%;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 14px;
}
.pc-search__input {
  height: 42px;
}
.pc-search__input:focus,
.section-head input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}
.pc-search__submit,
.preview-head button,
.pc-search__chips button {
  border: 0;
  border-radius: 6px;
  background: #2563eb;
  color: #fff;
  padding: 0 14px;
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  white-space: nowrap;
}
.pc-search__submit {
  width: 128px;
  min-width: 128px;
  padding: 0;
}
.pc-search__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px dashed #dbe3ef;
}
.pc-search__chips button {
  background: #eff6ff;
  color: #1d4ed8;
  height: auto;
  padding: 6px 10px;
  font-size: 12px;
}
.pc-search__preview {
  border-top: 1px solid #e2e8f0;
  padding-top: 12px;
}
.preview-head {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: space-between;
  align-items: center;
  color: #475569;
  font-size: 13px;
  margin-bottom: 8px;
}
.preview-head button {
  background: #f97316;
  height: auto;
  min-height: 30px;
  padding: 5px 10px;
}
.preview-list {
  display: grid;
  gap: 8px;
}
.preview-item button {
  width: 100%;
  text-align: left;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #fff;
  padding: 10px;
  cursor: pointer;
}
.preview-item button:hover {
  border-color: #93c5fd;
  background: #f8fbff;
}
.preview-item strong,
.preview-item span,
.preview-item small {
  display: block;
}
.preview-item strong {
  color: #0f172a;
  font-size: 14px;
  line-height: 1.35;
}
.preview-item span {
  color: #2563eb;
  margin-top: 4px;
  font-size: 12px;
}
.preview-item small {
  color: #64748b;
  margin-top: 3px;
}
.section-head {
  display: grid;
  grid-template-columns: 1fr 260px;
  gap: 16px;
  align-items: start;
  margin-bottom: 14px;
}
.section-head h3 {
  margin: 0;
  font-size: 18px;
}
.section-head p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
}
.venue-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.venue-tile {
  min-height: 150px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  padding: 12px;
  text-align: left;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.venue-tile:hover {
  border-color: #f97316;
  background: #fffaf5;
}
.venue-name {
  color: #ea580c;
  font-size: 18px;
  font-weight: 700;
}
.venue-desc {
  min-height: 36px;
  color: #475569;
  font-size: 13px;
  line-height: 1.35;
  display: -webkit-box;
  line-clamp: 2;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.venue-stat-line {
  color: #334155;
  font-size: 13px;
}
.muted {
  color: #64748b;
}
.rank-line {
  margin-top: auto;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.rank-line b {
  border: 1px solid #bfdbfe;
  background: #eff6ff;
  color: #1d4ed8;
  border-radius: 4px;
  padding: 2px 5px;
  font-size: 11px;
}
.hint {
  color: #64748b;
  margin: 8px 0;
}
.error {
  color: #dc2626;
}
@media (max-width: 900px) {
  .pc-top-nav {
    margin: -24px calc(50% - 50vw) 14px;
    padding: 0 20px;
  }
  .pc-top-nav__inner {
    height: auto;
    min-height: 72px;
    padding: 10px 0;
    align-items: flex-start;
    flex-direction: column;
  }
  .ghost-btn {
    align-self: flex-end;
  }
  .stats-strip,
  .venue-grid,
  .section-head {
    grid-template-columns: 1fr;
  }
  .stat-cell {
    border-right: 0;
    border-bottom: 1px solid #e2e8f0;
  }
  .stat-cell:last-child {
    border-bottom: 0;
  }
  .pc-search__main {
    grid-template-columns: 1fr;
    row-gap: 10px;
  }
  .pc-search__submit {
    width: 100%;
    min-width: 0;
    height: 42px;
  }
}
</style>
