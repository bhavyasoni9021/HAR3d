import { createApp } from './app.js';

const PORT = Number(process.env.PORT) || 8787;
const app = createApp();

app.listen(PORT, () => {
  console.log(`[har] API ready on http://localhost:${PORT}  (try /api/health)`);
});
