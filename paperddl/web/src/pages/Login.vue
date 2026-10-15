<script setup lang="ts">
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { login } from "../services/auth";

const route = useRoute();
const router = useRouter();

const username = ref("");
const password = ref("");
const loading = ref(false);
const errorMessage = ref("");

function getRedirectPath() {
  const redirect = route.query.redirect;
  if (typeof redirect === "string" && redirect.startsWith("/")) {
    return redirect;
  }
  return "/admin";
}

async function handleSubmit() {
  loading.value = true;
  errorMessage.value = "";
  try {
    await login(username.value.trim(), password.value);
    await router.replace(getRedirectPath());
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "登录失败";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="toolbar-card login-card">
    <header class="login-header">
      <h2>登录</h2>
    </header>

    <div v-if="errorMessage" class="alert alert-error">
      {{ errorMessage }}
    </div>

    <form class="login-form" @submit.prevent="handleSubmit">
      <div class="field">
        <label>用户名</label>
        <input v-model="username" autocomplete="username" type="text" required />
      </div>

      <div class="field">
        <label>密码</label>
        <input v-model="password" autocomplete="current-password" type="password" required />
      </div>

      <button class="submit-btn" type="submit" :disabled="loading">
        {{ loading ? "登录中..." : "登录" }}
      </button>

      <div class="login-footer">
        还没有账号？ <router-link to="/register">立即注册</router-link>
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

.alert-error {
  background: #fef2f2;
  color: #991b1b;
}
</style>
