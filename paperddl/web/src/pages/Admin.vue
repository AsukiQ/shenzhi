<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { fetchCategories, addConference } from '../services/api';
import type { Category } from '../types/conference';

const categories = ref<Category[]>([]);
const loading = ref(false);
const message = ref({ type: '', text: '' });

const form = ref({
  title: '',
  description: '',
  sub: '',
  rank: {
    ccf: 'A',
    core: '',
    thcpl: ''
  },
  confs: [
    {
      year: new Date().getFullYear(),
      id: '',
      link: '',
      timeline: [
        {
          abstract_deadline: '',
          deadline: ''
        }
      ]
    }
  ],
  source: 'manual'
});

onMounted(async () => {
  categories.value = await fetchCategories();
});

const handleSubmit = async () => {
  loading.value = true;
  message.value = { type: '', text: '' };
  try {
    // Basic validation
    if (!form.value.title || !form.value.sub) {
      throw new Error("名称和分类是必填项");
    }

    await addConference(form.value);
    message.value = { type: 'success', text: '会议添加成功！' };
    // Reset form partially
    form.value.title = '';
    form.value.description = '';
  } catch (e: any) {
    message.value = { type: 'error', text: e.message };
  } finally {
    loading.value = false;
  }
};
</script>

<template>
  <div class="toolbar-card admin-card">
    <header class="admin-header">
      <div class="admin-title">
        <h2>后台管理</h2>
        <div class="admin-subtitle">添加 / 更新会议数据</div>
      </div>
      <button type="button" class="back-btn" @click="$emit('close')">返回</button>
    </header>

    <div v-if="message.text" :class="['alert', `alert-${message.type}`]">
      {{ message.text }}
    </div>

    <form @submit.prevent="handleSubmit" class="admin-form">
      <section class="form-section">
        <h3>基本信息</h3>
        <div class="field">
          <label>会议名称 (缩写)</label>
          <input v-model="form.title" type="text" placeholder="例如：CVPR" required />
        </div>
        <div class="field">
          <label>详细描述</label>
          <textarea v-model="form.description" placeholder="例如：Conference on Computer Vision and Pattern Recognition"></textarea>
        </div>
        <div class="field">
          <label>学科分类</label>
          <select v-model="form.sub" required>
            <option value="">选择分类</option>
            <option v-for="c in categories" :key="c.sub" :value="c.sub">{{ c.name }} ({{ c.sub }})</option>
          </select>
        </div>
      </section>

      <section class="form-section">
        <h3>排名 (Rankings)</h3>
        <div class="grid-fields">
          <div class="field">
            <label>CCF 评级</label>
            <select v-model="form.rank.ccf">
              <option value="A">A</option>
              <option value="B">B</option>
              <option value="C">C</option>
              <option value="N">None</option>
            </select>
          </div>
          <div class="field">
            <label>CORE 评级</label>
            <input v-model="form.rank.core" type="text" placeholder="例如：A*" />
          </div>
        </div>
      </section>

      <section class="form-section">
        <h3>当前届次 ({{ form.confs[0].year }})</h3>
        <div class="field">
          <label>官方链接</label>
          <input v-model="form.confs[0].link" type="url" placeholder="https://..." />
        </div>
        <div class="grid-fields">
          <div class="field">
            <label>摘要截止日期</label>
            <input v-model="form.confs[0].timeline[0].abstract_deadline" type="text" placeholder="YYYY-MM-DD HH:mm:ss" />
          </div>
          <div class="field">
            <label>全文截稿日期</label>
            <input v-model="form.confs[0].timeline[0].deadline" type="text" placeholder="YYYY-MM-DD HH:mm:ss" />
          </div>
        </div>
        <p class="hint">格式：YYYY-MM-DD HH:mm:ss (本地时间)</p>
      </section>

      <button type="submit" :disabled="loading" class="submit-btn">
        {{ loading ? '保存中...' : '添加会议' }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.admin-card {
  border: 1px solid #f1f5f9;
}

.admin-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border-color);
}

.admin-title h2 {
  margin: 0;
  font-size: 1.05rem;
  color: var(--text-main);
}

.admin-subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.15rem;
}

.back-btn {
  border: 1px solid var(--border-color);
  background: #ffffff;
  border-radius: 999px;
  padding: 0.25rem 0.6rem;
  font-size: 0.8rem;
  cursor: pointer;
  color: #334155;
}

.back-btn:hover {
  border-color: #93c5fd;
  color: #2563eb;
}

.admin-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.form-section {
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 0.75rem;
}

.form-section h3 {
  margin: 0 0 0.6rem 0;
  color: var(--text-main);
  font-size: 0.9rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 0.6rem;
}

label {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-muted);
}

input, select, textarea {
  padding: 0.45rem 0.6rem;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  font-size: 0.85rem;
  background: #ffffff;
}

input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: #93c5fd;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.grid-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.submit-btn {
  background: var(--primary);
  color: white;
  border: none;
  padding: 0.65rem 0.8rem;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 700;
  cursor: pointer;
}

.submit-btn:hover:enabled {
  background: var(--primary-hover);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.alert {
  padding: 0.6rem 0.75rem;
  border-radius: 8px;
  margin-bottom: 0.75rem;
}

.alert-success { background: #ecfdf5; color: #065f46; }
.alert-error { background: #fef2f2; color: #991b1b; }
.hint { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }

@media (max-width: 720px) {
  .grid-fields {
    grid-template-columns: 1fr;
  }
}
</style>
