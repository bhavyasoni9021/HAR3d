# Human Activity Recognition with a Hybrid CNN-LSTM — interactive 3D case study

A three.js website plus a Node/Express backend for the SRM case study *Human Activity Recognition Using a Hybrid CNN-LSTM Network*.
The backend runs your **real trained model** (230,342 parameters) in pure JavaScript, so no Python or TensorFlow is needed.

## Run it

```bash
npm install
npm run dev
```

Open http://localhost:5173 (the API runs on port 8787 and is proxied automatically).
Needs Node 18.11 or newer and an internet connection for the Google Fonts (a system-font fallback is used offline).

| Command | What it does |
|---|---|
| `npm run dev` | API + Vite dev server with hot reload |
| `npm run build` then `npm start` | Production build served by Express on http://localhost:8787 |
| `npm test` | Backend tests (engine, simulator, dataset loader, HTTP API) |

## What is on the page

1. **Hero** – a 3D mannequin acts out the six activities beside nine streaming sensor traces.
2. **Problem and data** – motivation, objectives, the sliding-window animation, class balance, preprocessing.
3. **Model** – orbit-able 3D network. Every plate shows *real activations* from the backend for the window currently in the lab. Hover for shapes and parameters, click to fly to a layer.
4. **Sensor lab** – four ways to feed the model, all ported from the original Streamlit dashboard:
   - *Simulate*: pick an activity, noise, phone tilt and subject.
   - *Set values*: sliders for gravity direction, body-acceleration strength, **rotation (gyroscope) strength**, movement frequency and noise, with one-click typical-value presets (computed from the real data when available).
   - *Real samples*: filter the UCI test set by true activity, show only misclassified windows, random sample, slider.
   - *Your recording*: upload `.csv`/`.json`/`.npy` with 9 columns. 128 rows = one window; longer recordings are split into 50%-overlap windows with a clickable timeline, probabilities over time and time per activity. "Download example CSV" gives you a file in the right format.
5. **Results** – interactive 3D confusion matrix, per-class table, per-subject accuracy, training curves.
6. **Errors, roadmap, theory (CNN, LSTM, transfer learning, attention, GAN), team and references.**

## Honest notes

- The lab's generated windows are **synthetic** (gait harmonics, gravity orientation, noise). They are not UCI recordings. The page says so.
- Report numbers (92.77%, confusion matrix, per-class table) come from your report. Subjects 2 and 4 in the per-subject chart (89.1%, 87.0%) were read off Figure 3.5 and are marked approximate.
- **Real-data mode:** extract the UCI HAR dataset so that `data/UCI HAR Dataset/test/y_test.txt` exists here (or set `HAR_DATA_DIR`). On startup the server loads the test split, re-evaluates the model in the background and the page switches to *computed* metrics and a "Use a real test window" button. Check `GET /api/health` for progress.
- If you retrain: `python tools/export_weights.py outputs/best_model.keras outputs/norm.npz` rewrites `server/model/`.

## Layout

```
server/   Express API (app.js), lib/engine.js (CNN-LSTM forward pass), simulator.js, dataset.js, evaluation.js, data/report.json, model/ (weights), tests/
client/   Vite + three.js front end (index.html, src/*.js, styles.css, public/img)
tools/    export_weights.py (Keras → Node weights), h5mini.py (dependency-free HDF5 reader used for the first export)
```

API: `GET /api/health · /api/report · /api/model · /api/metrics · /api/simulate · /api/presets · /api/samples/index · /api/samples/random · /api/samples/:index` and `POST /api/predict · /api/predict-sequence · /api/synthesize`.
