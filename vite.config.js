import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.join(root, 'client'),
  publicDir: path.join(root, 'client', 'public'),
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8787', changeOrigin: true } },
  },
  build: { outDir: path.join(root, 'dist'), emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
