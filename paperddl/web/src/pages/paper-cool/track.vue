<template>
  <div class="track-page">
    <div class="page-header">
      <button @click="$router.back()" class="back-btn">Back</button>
      <div>
        <h2>{{ venueName }} Tracks</h2>
        <p>{{ totalPapers.toLocaleString() }} papers · {{ yearCount }} years · {{ trackCount }} tracks</p>
      </div>
    </div>

    <div class="toolbar">
      <div class="toolbar-field">
        <span>Track</span>
        <input v-model="trackFilter" placeholder="筛选 track 或年份" />
      </div>
      <div class="toolbar-field">
        <span>Paper</span>
        <input
          v-model="paperSearch"
          placeholder="在该会议内搜索论文"
          @keyup.enter="openVenueSearch"
        />
      </div>
      <button type="button" @click="openVenueSearch">搜索论文</button>
    </div>

    <p v-if="loading" class="hint">正在加载 track...</p>
    <p v-else-if="error" class="hint error">{{ error }}</p>
    <p v-else-if="Object.keys(filteredYearThemeMap).length === 0" class="hint">暂无论文数据</p>

    <div class="track-container">
      <div class="year-group" v-for="(themes, year) in filteredYearThemeMap" :key="year">
        <div class="year-title">
          <div>
            <span>{{ venueName }} {{ year }}</span>
            <small>{{ themes.length }} tracks</small>
          </div>
          <strong>{{ yearTotals[year]?.toLocaleString() || 0 }} papers</strong>
        </div>
        <div class="theme-grid">
          <button
            class="theme-block"
            v-for="theme in themes"
            :key="theme.track"
            type="button"
            @click="openTrack(String(year), theme.track)"
          >
            <span>{{ theme.track }}</span>
            <small>{{ theme.count.toLocaleString() }} papers</small>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { fetchPaperTracks } from "../../services/api";

interface ThemeInfo {
  track: string;
  count: number;
}

const route = useRoute();
const router = useRouter();
const venueName = ref(String(route.params.venue || "ICCV"));
const yearThemeMap = ref<Record<string, ThemeInfo[]>>({});
const trackFilter = ref("");
const paperSearch = ref("");
const loading = ref(false);
const error = ref("");

const totalPapers = computed(() => {
  return Object.values(yearThemeMap.value).flat().reduce((sum, item) => sum + item.count, 0);
});

const yearCount = computed(() => Object.keys(yearThemeMap.value).length);

const trackCount = computed(() => {
  const tracks = new Set<string>();
  Object.values(yearThemeMap.value).flat().forEach((item) => tracks.add(item.track));
  return tracks.size;
});

const yearTotals = computed(() => {
  const totals: Record<string, number> = {};
  for (const [year, themes] of Object.entries(yearThemeMap.value)) {
    totals[year] = themes.reduce((sum, item) => sum + item.count, 0);
  }
  return totals;
});

const filteredYearThemeMap = computed(() => {
  const q = trackFilter.value.trim().toLowerCase();
  if (!q) return yearThemeMap.value;

  const filtered: Record<string, ThemeInfo[]> = {};
  for (const [year, themes] of Object.entries(yearThemeMap.value)) {
    const matched = themes.filter((theme) => {
      return year.includes(q) || theme.track.toLowerCase().includes(q);
    });
    if (matched.length) filtered[year] = matched;
  }
  return filtered;
});

async function loadTracks() {
  loading.value = true;
  error.value = "";
  try {
    const tracks = await fetchPaperTracks(venueName.value);
    const tempMap: Record<string, ThemeInfo[]> = {};
    for (const item of tracks) {
      const year = String(item.year);
      if (!tempMap[year]) tempMap[year] = [];
      tempMap[year].push({ track: item.track, count: item.count });
    }
    yearThemeMap.value = Object.fromEntries(
      Object.entries(tempMap).sort(([a], [b]) => Number(b) - Number(a))
    );
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Track 加载失败";
    yearThemeMap.value = {};
  } finally {
    loading.value = false;
  }
}

onMounted(loadTracks);

watch(
  () => route.params.venue,
  async (venue) => {
    venueName.value = String(venue || "ICCV");
    await loadTracks();
  }
);

const openTrack = (year: string, track: string) => {
  router.push({
    name: "detail",
    params: {
      venue: venueName.value,
      year,
      track
    }
  });
};

const openVenueSearch = () => {
  const search = paperSearch.value.trim();
  router.push({
    name: "detail",
    params: {
      venue: venueName.value,
      year: "all",
      track: "all"
    },
    query: search ? { search } : {}
  });
};
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
.track-page {
  max-width: 1480px;
  margin: 0 auto;
  padding: 28px 24px 44px;
  font-family: system-ui;
}
.page-header {
  display: flex;
  align-items: center;
  gap: 18px;
  margin-bottom: 18px;
}
.page-header h2 {
  margin: 0 0 4px;
}
.page-header p {
  color: #666;
  margin: 0;
}
.back-btn {
  padding: 8px 14px;
  border: 1px solid #dbe3ef;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
}
.toolbar {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) minmax(260px, 1.3fr) auto;
  gap: 10px;
  margin-bottom: 22px;
  padding: 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
}
.toolbar-field {
  min-width: 0;
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 8px;
}
.toolbar-field span {
  color: #64748b;
  font-size: 13px;
  font-weight: 600;
}
.toolbar-field input {
  width: 100%;
  min-width: 0;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 9px 10px;
}
.toolbar-field input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}
.toolbar button {
  border: none;
  border-radius: 6px;
  background: #4CAF50;
  color: #fff;
  padding: 0 18px;
  cursor: pointer;
  white-space: nowrap;
}
.hint {
  color: #666;
  margin-bottom: 1rem;
}
.error {
  color: #d32f2f;
}
.track-container {
  display: flex;
  flex-direction: column;
  gap: 22px;
}
.year-group {
  width: 100%;
}
.year-title {
  font-size: 18px;
  font-weight: bold;
  color: #ea580c;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 10px;
}
.year-title div {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.year-title small {
  color: #777;
  font-weight: 400;
  font-size: 13px;
}
.year-title strong {
  color: #64748b;
  font-size: 14px;
  font-weight: 500;
}
.theme-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 12px;
}
.theme-block {
  min-height: 96px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #ea580c;
  font-size: 14px;
  line-height: 1.4;
  background: #fff;
  cursor: pointer;
  transition: all 0.2s;
  padding: 12px;
}
.theme-block span {
  font-size: 15px;
  font-weight: 600;
}
.theme-block small {
  color: #777;
  margin-top: 4px;
}
.theme-block:hover {
  border-color: #ff6600;
  background: #fff8f0;
}
@media (max-width: 900px) {
  .theme-grid {
    grid-template-columns: 1fr;
  }
  .toolbar {
    grid-template-columns: 1fr;
  }
  .toolbar button {
    padding: 8px 14px;
  }
  .toolbar-field {
    grid-template-columns: 1fr;
  }
  .year-title {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
