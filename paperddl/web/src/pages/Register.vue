<script setup lang="ts">
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { register } from "../services/auth";

const route = useRoute();
const router = useRouter();

const username = ref("");
const password = ref("");
const confirmPassword = ref("");
const loading = ref(false);
const errorMessage = ref("");

function getRedirectPath() {
  const redirect = route.query.redirect;
  if (typeof redirect === "string" && redirect.startsWith("/")) {
    return redirect;
  }
  return "/";
}

async function handleSubmit() {
  if (password.value !== confirmPassword.value) {
    errorMessage.value = "两次输入的密码不一致";
    return;
  }

  loading.value = true;
  errorMessage.value = "";
  try {
    await register(username.value.trim(), password.value);
    await router.replace(getRedirectPath());
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "注册失败";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="toolbar-card login-card">
    <header class="login-header">
      <h2>注册</h2>
      <div class="login-subtitle">创建新账户</div>
    </header>

    <div v-if="errorMessage" class="alert alert-error">
      {{ errorMessage }}
    </div>

    <form class="login-form" @submit.prevent="handleSubmit">
      <div class="field">
        <label>用户名</label>
        <input v-model="username" autocomplete="username" type="text" required placeholder="至少3位字符" />
      </div>

      <div class="field">
        <label>密码</label>
        <input v-model="password" autocomplete="new-password" type="password" required placeholder="至少6位字符" />
      </div>

      <div class="field">
        <label>确认密码</label>
        <input v-model="confirmPassword" autocomplete="new-password" type="password" required />
      </div>

      <button class="submit-btn" type="submit" :disabled="loading">
        {{ loading ? "正在注册..." : "提交注册" }}
      </button>

      <div class="login-footer">
        已有账号？ <router-link to="/login">立即登录</router-link>
      </div>
    </form>
  </div>
</template>

<style scoped>
.login-card {
  max-width: 420px;
  margin: 1.5rem auto;
}

.login-header {
  margin-bottom: 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border-color);
}

.login-header h2 {
  margin: 0;
  font-size: 1.05rem;
  color: var(--text-main);
}

.login-subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.15rem;
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.login-footer {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  text-align: center;
  color: var(--text-muted);
}

.login-footer a {
  color: var(--primary);
  text-decoration: none;
}

.login-footer a:hover {
  text-decoration: underline;
}

label {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-muted);
}

input {
  padding: 0.45rem 0.6rem;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  font-size: 0.85rem;
  background: #ffffff;
}

input:focus {
  outline: none;
  border-color: #93c5fd;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
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
  margin-top: 0.25rem;
  transition: all 0.2s;
}

.submit-btn:hover:enabled {
  background: var(--primary-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.submit-btn:active:enabled {
  transform: translateY(0);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.alert {
  padding: 0.65rem 0.8rem;
  border-radius: 6px;
  font-size: 0.8rem;
  margin-bottom: 0.75rem;
}

.alert-error {
  background-color: #fef2f2;
  color: #991b1b;
  border: 1px solid #fee2e2;
}
</style>
