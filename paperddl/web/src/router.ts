import { h } from "vue";
import { createRouter, createWebHashHistory } from "vue-router";
import AdminPage from "./pages/Admin.vue";
import LoginPage from "./pages/Login.vue";
import RegisterPage from "./pages/Register.vue";
import { loadCurrentUser } from "./services/auth";

const HomeRoute = {
  name: "HomeRoute",
  render: () => h("span")
};

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", name: "home", component: HomeRoute },
    { path: "/login", name: "login", component: LoginPage },
    { path: "/register", name: "register", component: RegisterPage },
    {
      path: "/admin",
      name: "admin",
      component: AdminPage,
      meta: { requiresAuth: true, requiresAdmin: true }
    },
    { path: "/paper-cool", name: "VenueAll", component: () => import("./pages/paper-cool/VenueAll.vue") },
    { path: "/paper-cool/:venue", name: "track", component: () => import("./pages/paper-cool/track.vue") },
    { path: "/paper-cool/:venue/:year/:track", name: "detail", component: () => import("./pages/paper-cool/detail.vue") }
  ]
});

router.beforeEach(async (to) => {
  if (to.meta.requiresAuth) {
    const user = await loadCurrentUser().catch(() => null);
    if (!user) {
      return {
        path: "/login",
        query: { redirect: to.fullPath }
      };
    }

    if (to.meta.requiresAdmin && user.role !== "admin") {
      return { path: "/" };
    }
  }

  if (to.name === "login") {
    const user = await loadCurrentUser().catch(() => null);
    if (user) {
      const redirect = typeof to.query.redirect === "string" ? to.query.redirect : "/admin";
      return redirect.startsWith("/") ? redirect : "/admin";
    }
  }
});

export default router;
