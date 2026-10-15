import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import "./style.css";

router.afterEach((to) => {
	const enabled = String(to.path || "").startsWith("/paper-cool");
	document.body.classList.toggle("paper-cool-mode", enabled);
});

createApp(App).use(router).mount("#app");
