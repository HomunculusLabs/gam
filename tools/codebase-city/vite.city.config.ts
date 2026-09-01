import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/city-runtime/",
  publicDir: false,
  plugins: [react()],
  build: {
    outDir: "public/city-runtime",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      input: "client/city-entry.tsx",
      output: {
        entryFileNames: "city.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
