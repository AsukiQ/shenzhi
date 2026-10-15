<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { fetchCategories, fetchConferences } from "./services/api";
import { currentUser, loadCurrentUser, logout } from "./services/auth";
import {
  addFavorite,
  fetchFavoriteIds,
  removeFavorite
} from "./services/favorites";
import type { Category, Conference, ConferenceQuery } from "./types/conference";
import ConferenceCard from "./components/ConferenceCard.vue";


const categories = ref<Category[]>([]);
const conferences = ref<Conference[]>([]);
const query = ref<ConferenceQuery>({
  sub: [] as string[],
  q: ""
});
const loading = ref(false);
const route = useRoute();
const router = useRouter();
const isHome = computed(() => route.name === "home");

function goLogin() {
  router.push("/login");
}

function goHome() {
  router.push("/");
}

function goPaperCool() {
  router.push("/paper-cool");
}

const searchPanelRef = ref<HTMLElement | null>(null);

function scrollToTop() {
  if (!searchPanelRef.value) return;

  const top =
    searchPanelRef.value.getBoundingClientRect().top +
    window.scrollY -
    70;

  window.scrollTo({
    top,
    behavior: "smooth"
  });
}

async function handleLogout() {
  await logout();
  userMenuOpen.value = false;
  router.push("/");
}

type RankValue = "any" | "A*" | "A" | "B" | "C" | "N";

const ccfRank = ref<Exclude<RankValue, "A*">>("any");
const coreRank = ref<RankValue>("any");
const thcplRank = ref<Exclude<RankValue, "A*">>("any");
const yearsSelected = ref<number[]>([]);
const userMenuOpen = ref(false);
const favoriteIds = ref<Set<string>>(new Set());
const showFavoritesOnly = ref(false);
const pendingFavoriteIds = ref<Set<string>>(new Set());
const favoritesMenuLabel = computed(() =>
  showFavoritesOnly.value ? "全部会议" : "收藏会议"
);
const resultsHeading = computed(() =>
  showFavoritesOnly.value
    ? `My favorites (${visibleConferences.value.length})`
    : `Recommended for you (${visibleConferences.value.length})`
);

const heroTitle = ref("快速找到合适的投稿目标");

const availableYears = computed(() => {
  const years = new Set<number>();
  for (const conf of conferences.value) {
    const anyConf = conf as any;
    const y = Number(anyConf?.displayYear ?? conf.confs?.[0]?.year);
    if (Number.isFinite(y)) years.add(y);
  }
  return Array.from(years).sort((a, b) => b - a);
});

function getRank(conf: Conference, system: "ccf" | "core" | "thcpl"): string {
  const rank = conf.rank || ({} as any);
  if (system === "ccf") return (rank as any).ccf || "N";
  if (system === "core") return (rank as any).core || "N";
  return (rank as any).thcpl || (rank as any).thc || "N";
}

function getConferenceFavoriteId(conf: Conference): string {
  const anyConf = conf as any;
  return String(
    anyConf?.edition?.id ||
    conf.confs?.[0]?.id ||
    `${conf.title}-${anyConf?.displayYear || conf.confs?.[0]?.year || ""}`
  );
}

const ccfRankOptions = [
  { value: "any", label: "All" },
  { value: "A", label: "A" },
  { value: "B", label: "B" },
  { value: "C", label: "C" },
  { value: "N", label: "Non" }
] as const;

const coreRankOptions = [
  { value: "any", label: "All" },
  { value: "A*", label: "A*" },
  { value: "A", label: "A" },
  { value: "B", label: "B" },
  { value: "C", label: "C" },
  { value: "N", label: "Non" }
] as const;

const thcplRankOptions = [
  { value: "any", label: "All" },
  { value: "A", label: "A" },
  { value: "B", label: "B" },
  { value: "C", label: "C" },
  { value: "N", label: "Non" }
] as const;

const visibleConferences = computed(() => {
  const selectedYears = new Set(yearsSelected.value);
  return conferences.value.filter((conf) => {
    if (showFavoritesOnly.value && !favoriteIds.value.has(getConferenceFavoriteId(conf))) {
      return false;
    }

    if (selectedYears.size) {
      const anyConf = conf as any;
      const year = Number(anyConf?.displayYear ?? conf.confs?.[0]?.year);
      if (!Number.isFinite(year) || !selectedYears.has(year)) return false;
    }

    if (ccfRank.value !== "any") {
      const value = getRank(conf, "ccf");
      if ((value || "N") !== ccfRank.value) return false;
    }

    if (coreRank.value !== "any") {
      const value = getRank(conf, "core");
      if ((value || "N") !== coreRank.value) return false;
    }

    if (thcplRank.value !== "any") {
      const value = getRank(conf, "thcpl");
      if ((value || "N") !== thcplRank.value) return false;
    }

    return true;
  });
});

const pageSize = 15;
const currentPage = ref(1);

const totalPages = computed(() => {
  return Math.max(1, Math.ceil(visibleConferences.value.length / pageSize));
});

const paginatedConferences = computed(() => {
  const start = (currentPage.value - 1) * pageSize;
  return visibleConferences.value.slice(start, start + pageSize);
});

const paginationStart = computed(() => {
  if (visibleConferences.value.length === 0) return 0;
  return (currentPage.value - 1) * pageSize + 1;
});

const paginationEnd = computed(() => {
  return Math.min(currentPage.value * pageSize, visibleConferences.value.length);
});

const pageNumbers = computed(() => {
  const total = totalPages.value;
  const current = currentPage.value;
  const delta = 2;

  const start = Math.max(1, current - delta);
  const end = Math.min(total, current + delta);

  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
});

function goToPage(page: number) {
  const nextPage = Math.min(Math.max(page, 1), totalPages.value);

  if (nextPage === currentPage.value) return;

  currentPage.value = nextPage;

  nextTick(() => {
    scrollToTop();
  });
}

const loadData = async () => {
  loading.value = true;
  try {
    const [cats, confs] = await Promise.all([
      fetchCategories(),
      fetchConferences(query.value)
    ]);
    categories.value = cats;
    const nowMs = Date.now();
    const parseConfDeadline = (conf: Conference) => {
      const edition = conf.confs?.[0];
      const candidate = conf.nextDeadline || edition?.timeline?.[0]?.deadline;
      const tz = edition?.timezone || conf.timezone;
      if (!candidate || candidate.toUpperCase() === "TBD") return null;

      const parts = candidate.split(" ");
      if (parts.length === 2 && tz) {
        const tzUpper = tz.toUpperCase();
        let tzOffset = 0;
        if (tzUpper === "AOE") {
          tzOffset = -12;
        } else {
          const match = tz.match(/UTC([+-]\d+)/i);
          if (match) tzOffset = parseInt(match[1], 10);
        }
        const [datePart, timePart] = parts;
        const [y, m, d] = datePart.split("-").map(Number);
        const [hh, mm, ss] = timePart.split(":").map(Number);
        return Date.UTC(y, m - 1, d, hh - tzOffset, mm, ss);
      }
      const parsed = Date.parse(candidate.replace(" ", "T"));
      return Number.isNaN(parsed) ? null : parsed;
    };
    const isExpired = (deadline: number | null) => deadline !== null && deadline <= nowMs;
    const isUrgent = (deadline: number | null, expired: boolean) => {
      if (deadline === null || expired) return false;
      const diff = deadline - nowMs;
      return diff > 0 && diff < 24 * 60 * 60 * 1000;
    };
    const expandedConfs = confs.flatMap((conf) => {
      if (!conf.confs || conf.confs.length === 0) {
        return [conf];
      }
      return conf.confs.map((edition) => {
        const editionDeadline = edition.timeline?.[0]?.deadline || conf.nextDeadline;
        return {
          ...conf,
          confs: [edition],
          nextDeadline: editionDeadline,
          edition,
          displayYear: edition.year
        } as Conference;
      });
    });
    conferences.value = expandedConfs
      .slice()
      .sort((a: Conference, b: Conference) => {
        const aTime = parseConfDeadline(a);
        const bTime = parseConfDeadline(b);
        const aExpired = isExpired(aTime);
        const bExpired = isExpired(bTime);
        if (aExpired && !bExpired) return 1;
        if (!aExpired && bExpired) return -1;
        const aUrgent = isUrgent(aTime, aExpired);
        const bUrgent = isUrgent(bTime, bExpired);
        if (aUrgent && !bUrgent) return -1;
        if (!aUrgent && bUrgent) return 1;
        if (aTime === null && bTime === null) return 0;
        if (aTime === null) return 1;
        if (bTime === null) return -1;
        return aTime - bTime;
      });
  } catch (e) {
    console.error("Failed to fetch data:", e);
  } finally {
    loading.value = false;
  }
};

async function loadFavorites() {
  if (!currentUser.value) {
    favoriteIds.value = new Set();
    showFavoritesOnly.value = false;
    pendingFavoriteIds.value = new Set();
    return;
  }

  try {
    const ids = await fetchFavoriteIds();
    favoriteIds.value = new Set(ids);
  } catch (error) {
    console.error("Failed to load favorites:", error);
  }
}

async function handleToggleFavorite(conf: Conference) {
  if (!currentUser.value) {
    window.alert("请先登录后收藏会议");
    router.push("/login");
    return;
  }

  const conferenceId = getConferenceFavoriteId(conf);
  if (!conferenceId) return;

  if (pendingFavoriteIds.value.has(conferenceId)) return;

  const nextPending = new Set(pendingFavoriteIds.value);
  nextPending.add(conferenceId);
  pendingFavoriteIds.value = nextPending;

  const wasFavorite = favoriteIds.value.has(conferenceId);
  const nextFavorites = new Set(favoriteIds.value);

  if (wasFavorite) nextFavorites.delete(conferenceId);
  else nextFavorites.add(conferenceId);
  favoriteIds.value = nextFavorites;

  try {
    if (wasFavorite) {
      await removeFavorite(conferenceId);
    } else {
      await addFavorite(conferenceId);
    }
  } catch (error) {
    console.error("Failed to toggle favorite:", error);
    const rollback = new Set(favoriteIds.value);
    if (wasFavorite) rollback.add(conferenceId);
    else rollback.delete(conferenceId);
    favoriteIds.value = rollback;
    window.alert("收藏操作失败，请稍后重试");
  } finally {
    const finalPending = new Set(pendingFavoriteIds.value);
    finalPending.delete(conferenceId);
    pendingFavoriteIds.value = finalPending;
  }
}

function toggleFavoritesOnly() {
  if (!currentUser.value) {
    window.alert("请先登录后查看收藏会议");
    router.push("/login");
    return;
  }

  showFavoritesOnly.value = !showFavoritesOnly.value;
  userMenuOpen.value = false;
}

function handleSearchInput(e: Event) {
  const target = e.target as HTMLInputElement;
  query.value.q = target.value;
}

function handleSearchSubmit() {
  loadData();
}

function toggleYear(year: number) {
  const set = new Set(yearsSelected.value);
  if (set.has(year)) set.delete(year);
  else set.add(year);
  yearsSelected.value = Array.from(set).sort((a, b) => b - a);
}

function clearFilters() {
  query.value.sub = [];
  yearsSelected.value = [];
  ccfRank.value = "any";
  coreRank.value = "any";
  thcplRank.value = "any";
}

function toggleUserMenu() {
  userMenuOpen.value = !userMenuOpen.value;
}

function handleDocumentClick(event: MouseEvent) {
  const target = event.target as HTMLElement | null;
  if (!target?.closest(".user-menu")) {
    userMenuOpen.value = false;
  }
}

onMounted(async () => {
  try {
    await loadCurrentUser();
  } catch {
    // Auth state is optional on public pages.
  }
  await Promise.all([loadData(), loadFavorites()]);
  document.addEventListener("click", handleDocumentClick);
});

onBeforeUnmount(() => {
  document.removeEventListener("click", handleDocumentClick);
});

watch(
  query,
  () => {
    loadData();
  },
  { deep: true }
);

watch(
  visibleConferences,
  () => {
    currentPage.value = 1;
  }
);

watch(
  currentUser,
  () => {
    loadFavorites();
  }
);

</script>


<template>
  <div class="app-shell">
    <template v-if="isHome">
      <header class="top-nav">
        <div class="top-nav__inner">
          <button class="brand-mark" type="button" @click="goHome">
            <span class="brand-mark__title">PaperDDL</span>
            <span class="brand-mark__subtitle">Conference Deadlines</span>
          </button>

          <div class="top-nav__actions">
            <button class="nav-ghost-btn" type="button" @click="goPaperCool">
              论文库
            </button>

            <button
              v-if="!currentUser"
              class="nav-ghost-btn"
              type="button"
              @click="goLogin"
            >
              登录
            </button>

            <div v-else class="user-menu">
              <button class="user-menu__trigger" type="button" @click.stop="toggleUserMenu">
                <span class="user-menu__name">{{ currentUser.username }}</span>
                <span class="user-menu__caret">{{ userMenuOpen ? "^" : "v" }}</span>
              </button>
              <div v-if="userMenuOpen" class="user-menu__panel">
                <button
                  class="user-menu__item"
                  type="button"
                  @click="toggleFavoritesOnly"
                >
                  {{ favoritesMenuLabel }}
                </button>
                <button class="user-menu__item" type="button" @click="handleLogout">
                  退出登录
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main class="home-page">
        <section class="hero-banner">
          <div class="hero-banner__overlay">
            <p class="hero-banner__eyebrow">PaperDDL</p>
            <h1 class="hero-banner__title">{{ heroTitle }}</h1>
          </div>
        </section>

        <section ref="searchPanelRef" class="search-panel">
          <form class="search-panel__form" @submit.prevent="handleSearchSubmit">
            <input
              class="search-panel__input"
              type="text"
              :value="query.q"
              placeholder="搜索会议名称、简称或关键词"
              @input="handleSearchInput"
            />
            <button class="search-panel__button" type="submit">搜索</button>
          </form>
        </section>

        <section class="content-layout">
          <aside class="filter-sidebar">
            <div class="filter-sidebar__header">
              <h2>筛选条件</h2>
              <button class="filter-clear-btn" type="button" @click="clearFilters">
                清除
              </button>
            </div>

            <div class="filter-group">
              <div class="filter-group__title">研究方向</div>
              <div class="chip-list">
                <label v-for="cat in categories" :key="cat.sub" class="chip-check">
                  <input v-model="query.sub" type="checkbox" :value="cat.sub" />
                  <span>{{ cat.name }}</span>
                </label>
              </div>
            </div>

            <div v-if="availableYears.length" class="filter-group">
              <div class="filter-group__title">年份</div>
              <div class="chip-list chip-list--compact">
                <label v-for="y in availableYears" :key="y" class="chip-check">
                  <input
                    type="checkbox"
                    :checked="yearsSelected.includes(y)"
                    @change="toggleYear(y)"
                  />
                  <span>{{ y }}</span>
                </label>
              </div>
            </div>

            <div class="filter-group">
              <div class="filter-group__title">CCF</div>
              <select v-model="ccfRank" class="filter-select" aria-label="CCF rank">
                <option v-for="opt in ccfRankOptions" :key="opt.value" :value="opt.value">
                  {{ opt.label }}
                </option>
              </select>
            </div>

            <div class="filter-group">
              <div class="filter-group__title">CORE</div>
              <select v-model="coreRank" class="filter-select" aria-label="CORE rank">
                <option v-for="opt in coreRankOptions" :key="opt.value" :value="opt.value">
                  {{ opt.label }}
                </option>
              </select>
            </div>

            <div class="filter-group">
              <div class="filter-group__title">THCPL</div>
              <select v-model="thcplRank" class="filter-select" aria-label="THCPL rank">
                <option v-for="opt in thcplRankOptions" :key="opt.value" :value="opt.value">
                  {{ opt.label }}
                </option>
              </select>
            </div>
          </aside>

          <section class="results-panel">
            <div class="results-panel__header">
              <div>
                <p class="results-panel__caption">为你推荐</p>
                <h2
                  class="results-panel__title"
                >
                  {{ resultsHeading }}
                </h2>
              </div>
            </div>

            <div v-if="loading" class="state-panel">加载中...</div>
            <div v-else-if="visibleConferences.length === 0" class="state-panel">
              未找到相关会议。请尝试调整筛选条件或搜索关键词。;
            </div>
            <div v-else class="conference-list">
              <ConferenceCard
                v-for="conf in paginatedConferences"
                :key="`${conf.title}-${(conf as any).displayYear || conf.confs?.[0]?.year || ''}`"
                :conference="conf"
                :highlight="query.q || ''"
                :is-favorite="favoriteIds.has(getConferenceFavoriteId(conf))"
                :favorite-pending="pendingFavoriteIds.has(getConferenceFavoriteId(conf))"
                @toggle-favorite="handleToggleFavorite"
              />
            </div>
            <div v-if="visibleConferences.length > 0" class="pagination-summary">
              当前显示 {{ paginationStart }}-{{ paginationEnd }} 条，共 {{ visibleConferences.length }} 条
            </div>

            <nav v-if="totalPages > 1" class="pagination" aria-label="会议列表分页">
              <button
                class="pagination__btn"
                type="button"
                :disabled="currentPage === 1"
                @click="goToPage(currentPage - 1)"
              >
                上一页
              </button>

              <button
                v-for="page in pageNumbers"
                :key="page"
                class="pagination__btn"
                :class="{ 'pagination__btn--active': page === currentPage }"
                type="button"
                @click="goToPage(page)"
              >
                {{ page }}
              </button>

              <button
                class="pagination__btn"
                type="button"
                :disabled="currentPage === totalPages"
                @click="goToPage(currentPage + 1)"
              >
                下一页
              </button>
            </nav>
          </section>
        </section>
      </main>
      <button
        class="back-to-top"
        type="button"
        @click="scrollToTop"
        aria-label="返回顶部"
        title="返回顶部"
      >
        <svg
          class="back-to-top__icon"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path
            d="M12 19V5"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
          />
          <path
            d="M6 11L12 5L18 11"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </button>
    </template>

    <RouterView v-else v-slot="{ Component }">
      <div :class="{ 'admin-route-shell': route.name === 'admin' }">
        <component :is="Component" @close="goHome" />
      </div>
    </RouterView>
  </div>
</template>

<style lang="css">
  .pagination-summary {
    margin-top: 18px;
    text-align: center;
    font-size: 14px;
    color: #64748b;
  }

  .pagination {
    margin-top: 14px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .pagination__btn {
    min-width: 38px;
    height: 36px;
    padding: 0 12px;
    border: 1px solid rgba(203, 213, 225, 0.9);
    border-radius: 999px;
    background: #ffffff;
    color: #334155;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.18s ease;
  }

  .pagination__btn:hover:not(:disabled) {
    border-color: #2563eb;
    color: #2563eb;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12);
  }

  .pagination__btn--active {
    border-color: #2563eb;
    background: #2563eb;
    color: #ffffff;
  }

  .pagination__btn:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }
</style>
