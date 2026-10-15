import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
	plugins: [vue()],
	server: {
		proxy: {
			"/api/retrieval": {
                target: "http://127.0.0.1:8080",
                changeOrigin: true,
                rewrite: (url) => url.replace(/^\/api\/retrieval/, "")
            },
            "/api/knowledge": {
                target: "http://127.0.0.1:8080",
                changeOrigin: true,
                rewrite: (url) => url.replace(/^\/api\/knowledge/, "/knowledge")
            },
            "/api": {
				target: "http://localhost:5000",
				changeOrigin: true
			}
		}
	}
});
