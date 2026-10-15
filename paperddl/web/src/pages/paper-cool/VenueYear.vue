<template>
  <div class="pc-wrap">
    <div class="top-bar">
      <button @click="$router.back()">← Back</button>
      <h2>{{ venue }} - All Years</h2>
    </div>

    <div class="year-list">
      <div v-for="y in years" :key="y.year" class="year-item" @click="goYear(y.year)">
        <h3>{{ venue }} {{ y.year }}</h3>
        <p>{{ y.date }}</p>
        <p>{{ y.place }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchConferences } from '../../services/api'

const route = useRoute()
const router = useRouter()
const venue = ref('')
const years = ref([])

onMounted(async () => {
  venue.value = route.params.venue
  const data = await fetchConferences({ sub: [], q: '' })
  const conf = data.find(i => i.title.split(' ')[0] === venue.value)
  years.value = conf?.confs || []
})

const goYear = (year) => router.push(`/paper-cool/${venue.value}/${year}`)
</script>

<style scoped>
.pc-wrap{max-width:1200px;margin:0 auto;padding:2rem;font-family:system-ui}
.top-bar{display:flex;align-items:center;gap:1rem;margin-bottom:2rem}
.year-list{display:grid;grid-template-columns:repeat(2,1fr);gap:1.5rem}
.year-item{border:1px solid #eee;padding:1rem;border-radius:8px;cursor:pointer}
.year-item h3{color:#ff6600;margin:0 0 8px 0}
.year-item p{margin:4px 0;color:#666}
</style>
