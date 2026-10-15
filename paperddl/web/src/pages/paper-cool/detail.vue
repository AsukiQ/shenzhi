<template>
  <div class="detail-page">
    <header class="result-header">
      <button class="ghost-btn" type="button" @click="$router.back()">返回</button>
      <div class="result-title">
        <h2>{{ pageTitle }}</h2>
        <p>{{ resultSummary }}</p>
      </div>
      <button class="toggle-btn" type="button" @click="showAbstract = !showAbstract">
        {{ showAbstract ? '隐藏摘要' : '显示摘要' }}
      </button>
    </header>

    <section class="filter-panel">
      <div class="search-row">
        <input
          v-model="localSearch"
          placeholder="输入关键词后自动从后端查询"
          @input="scheduleSearch"
          @keyup.enter="applySearchNow"
        />
        <button type="button" @click="applySearchNow">立即搜索</button>
        <button v-if="localSearch || searchKey" class="plain-btn" type="button" @click="clearSearch">清空</button>
      </div>
      <div class="active-scope">
        <span v-if="venue && venue !== 'global'">会议：{{ venue }}</span>
        <span v-if="year && year !== 'global' && year !== 'all'">年份：{{ year }}</span>
        <span v-if="track && track !== 'global' && track !== 'all'">Track：{{ track }}</span>
        <span v-if="!hasScope">范围：全库</span>
      </div>
    </section>

    <p v-if="loading && paperList.length === 0" class="status">正在从后端加载论文...</p>
    <p v-else-if="error" class="status error">{{ error }}</p>

    <section v-else-if="paperList.length === 0" class="no-data">
      <h4>未找到相关论文</h4>
      <p>可以调整关键词，或返回会议页选择其他年份和 track。</p>
    </section>

    <section v-else class="paper-list">
      <article v-for="paper in paperList" :key="paper.id" class="paper-row">
        <div class="paper-main">
          <div class="paper-title-line">
            <h3>{{ paper.title }}</h3>
            <span>{{ paper.conference }} {{ paper.year }}</span>
          </div>
          <p class="authors">{{ paper.authors.join(', ') || 'Unknown authors' }}</p>
          <div class="meta-line">
            <span v-if="paper.track">{{ paper.track }}</span>
            <span v-if="paper.type">{{ paper.type }}</span>
            <span v-if="paper.publisher">{{ paper.publisher }}</span>
            <a v-if="paper.doi" :href="`https://doi.org/${paper.doi}`" target="_blank">DOI</a>
          </div>
          <p v-if="showAbstract && paper.abstract" class="abstract">{{ paper.abstract }}</p>
          <div v-if="paper.keywords.length" class="keywords">
            <span v-for="kw in paper.keywords" :key="kw">{{ kw }}</span>
          </div>
        </div>
        <div class="paper-actions">
          <button type="button" @click="openUrl(paper.pdf_url)">PDF</button>
          <button v-if="paper.biburl" type="button" class="plain-btn" @click="openUrl(paper.biburl)">Bib</button>
          <button type="button" class="plain-btn" @click="copyTitle(paper.title)">复制</button>
        </div>
      </article>
    </section>

    <div v-if="paperList.length < total && !error" class="load-more">
      <button type="button" :disabled="loading" @click="loadMore">
        {{ loading ? '加载中...' : `加载更多 (${paperList.length}/${total})` }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { fetchPapers } from "../../services/api";
import type { PaperQuery, PaperRecord } from "../../types/paper";

const route = useRoute();
const router = useRouter();
const pageSize = 50;

const venue = ref("");
const year = ref("");
const track = ref("");
const paperList = ref<PaperRecord[]>([]);
const total = ref(0);
const searchKey = ref("");
const localSearch = ref("");
const showAbstract = ref(true);
const loading = ref(false);
const error = ref("");
let searchTimer: number | undefined;
let requestSeq = 0;

const hasScope = computed(() => {
  return Boolean(
    (venue.value && venue.value !== "global") ||
    (year.value && year.value !== "global" && year.value !== "all") ||
    (track.value && track.value !== "global" && track.value !== "all")
  );
});

const pageTitle = computed(() => {
  if (searchKey.value && hasScope.value) return `在当前范围搜索「${searchKey.value}」`;
  if (searchKey.value) return `全库搜索「${searchKey.value}」`;
  if (venue.value && venue.value !== "global" && year.value === "all") return `${venue.value} 全部论文`;
  return `${venue.value} ${year.value} - ${track.value}`;
});

const resultSummary = computed(() => {
  const loaded = paperList.value.length.toLocaleString();
  const count = total.value.toLocaleString();
  return `后端匹配 ${count} 篇，已加载 ${loaded} 篇`;
});

function syncFromRoute() {
  venue.value = String(route.params.venue || "global");
  year.value = String(route.params.year || "global");
  track.value = String(route.params.track || "global").trim();
  searchKey.value = typeof route.query.search === "string" ? route.query.search.trim() : "";
  localSearch.value = searchKey.value;
}

function buildQuery(offset = 0): PaperQuery {
  const query: PaperQuery = { limit: pageSize, offset };
  if (searchKey.value) query.q = searchKey.value;
  if (venue.value && venue.value !== "global") query.conference = venue.value;
  if (year.value && year.value !== "global" && year.value !== "all") query.year = year.value;
  if (track.value && track.value !== "global" && track.value !== "all") query.track = track.value;
  return query;
}

async function loadPapers(reset = true) {
  const seq = ++requestSeq;
  const offset = reset ? 0 : paperList.value.length;
  loading.value = true;
  error.value = "";
  if (reset) {
    paperList.value = [];
    total.value = 0;
  }

  try {
    const result = await fetchPapers(buildQuery(offset));
    if (seq !== requestSeq) return;
    paperList.value = reset ? result.items : paperList.value.concat(result.items);
    total.value = result.total;
  } catch (err) {
    if (seq !== requestSeq) return;
    error.value = err instanceof Error ? err.message : "论文加载失败";
    if (reset) paperList.value = [];
    total.value = 0;
  } finally {
    if (seq === requestSeq) loading.value = false;
  }
}

function replaceSearch(search: string) {
  router.replace({
    name: "detail",
    params: {
      venue: venue.value || "global",
      year: year.value || "global",
      track: track.value || "global"
    },
    query: search ? { search } : {}
  });
}

function scheduleSearch() {
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => {
    replaceSearch(localSearch.value.trim());
  }, 350);
}

function applySearchNow() {
  window.clearTimeout(searchTimer);
  replaceSearch(localSearch.value.trim());
}

function clearSearch() {
  window.clearTimeout(searchTimer);
  localSearch.value = "";
  replaceSearch("");
}

async function loadMore() {
  await loadPapers(false);
}

function openUrl(url: string) {
  if (!url) {
    alert("暂无链接");
    return;
  }
  window.open(url, "_blank");
}

function copyTitle(text: string) {
  navigator.clipboard
    .writeText(text)
    .then(() => alert("复制成功"))
    .catch(() => alert("复制失败"));
}

onMounted(async () => {
  syncFromRoute();
  await loadPapers(true);
});

watch(
  () => [route.params.venue, route.params.year, route.params.track, route.query.search],
  async () => {
    syncFromRoute();
    await loadPapers(true);
  }
);

onBeforeUnmount(() => {
  window.clearTimeout(searchTimer);
});
</script>

<style scoped>
.detail-page {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px 20px 40px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.result-header {
  display: flex;
  align-items: center;
  gap: 14px;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 14px;
  margin-bottom: 14px;
}
.result-title {
  flex: 1;
}
.result-title h2 {
  margin: 0 0 4px;
  font-size: 22px;
  line-height: 1.35;
}
.result-title p {
  margin: 0;
  color: #64748b;
  font-size: 13px;
}
.ghost-btn,
.toggle-btn,
.plain-btn,
.paper-actions button,
.load-more button,
.search-row button {
  border: 0;
  border-radius: 6px;
  cursor: pointer;
}
.ghost-btn,
.plain-btn {
  border: 1px solid #dbe3ef;
  background: #fff !important;
  color: #334155 !important;
  padding: 8px 12px;
}
.toggle-btn,
.search-row button,
.load-more button {
  background: #2563eb;
  color: #fff;
  padding: 8px 12px;
}
.filter-panel {
  border: 1px solid #e2e8f0;
  background: #fff;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 14px;
}
.search-row {
  display: flex;
  flex-wrap: wrap;
  margin: -4px;
}
.search-row > * {
  margin: 4px;
}
.search-row input {
  flex: 1 1 520px;
  min-width: 240px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 10px 12px;
}
.search-row input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}
.active-scope {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.active-scope span {
  background: #f1f5f9;
  color: #475569;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
}
.status {
  color: #64748b;
  padding: 14px 0;
}
.error {
  color: #dc2626;
}
.no-data {
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  background: #fff;
  padding: 28px;
  text-align: center;
  color: #64748b;
}
.paper-list {
  display: grid;
  gap: 10px;
}
.paper-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  padding: 14px;
}
.paper-title-line {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}
.paper-title-line h3 {
  margin: 0;
  font-size: 16px;
  line-height: 1.4;
}
.paper-title-line span {
  color: #2563eb;
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
}
.authors {
  color: #475569;
  font-size: 13px;
  margin: 8px 0;
}
.meta-line {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}
.meta-line span,
.meta-line a,
.keywords span {
  background: #f1f5f9;
  color: #334155;
  border-radius: 4px;
  padding: 3px 7px;
  font-size: 12px;
  text-decoration: none;
}
.abstract {
  color: #334155;
  font-size: 13px;
  line-height: 1.6;
  margin: 8px 0;
}
.keywords {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.paper-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 76px;
}
.paper-actions button {
  background: #f97316;
  color: #fff;
  padding: 7px 10px;
}
.load-more {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}
.load-more button:disabled {
  opacity: 0.6;
  cursor: wait;
}
@media (max-width: 860px) {
  .result-header,
  .search-row,
  .paper-row,
  .paper-title-line {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
  .paper-actions {
    flex-direction: row;
    flex-wrap: wrap;
  }
  .paper-title-line span {
    white-space: normal;
  }
}
</style>
