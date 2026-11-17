"""Re-export a retrained Keras model for the Node backend.

Usage (inside your original project venv, which has TensorFlow):
    python tools/export_weights.py path/to/outputs/best_model.keras path/to/outputs/norm.npz
Writes server/model/{weights.bin,manifest.json,norm.json}.
"""
import json, sys, numpy as np
from pathlib import Path
from tensorflow import keras

model_path, norm_path = sys.argv[1], sys.argv[2]
out = Path(__file__).resolve().parent.parent / "server" / "model"
m = keras.models.load_model(model_path)
L = {l.name: l for l in m.layers}

def layer(prefix):
    return next(l for n, l in L.items() if n.startswith(prefix) and not n.startswith(prefix + "_") or n == prefix)

convs = [l for l in m.layers if l.__class__.__name__ == "Conv1D"]
bns = [l for l in m.layers if l.__class__.__name__ == "BatchNormalization"]
lstms = [l for l in m.layers if l.__class__.__name__ == "LSTM"]
dens = [l for l in m.layers if l.__class__.__name__ == "Dense"]
assert len(convs) == 2 and len(bns) == 2 and len(lstms) == 2 and len(dens) == 2, "unexpected architecture"

T = {}
for i, (c, b) in enumerate(zip(convs, bns), 1):
    T[f"conv{i}.w"], T[f"conv{i}.b"] = c.get_weights()
    T[f"bn{i}.gamma"], T[f"bn{i}.beta"], T[f"bn{i}.mean"], T[f"bn{i}.var"] = b.get_weights()
for i, l in enumerate(lstms, 1):
    T[f"lstm{i}.W"], T[f"lstm{i}.U"], T[f"lstm{i}.b"] = l.get_weights()
for i, d in enumerate(dens, 1):
    T[f"dense{i}.w"], T[f"dense{i}.b"] = d.get_weights()

manifest, chunks, off = {}, [], 0
for k, a in T.items():
    a = a.astype("<f4")
    manifest[k] = {"shape": list(a.shape), "offset": off, "length": int(a.size)}
    chunks.append(a.tobytes()); off += a.size
out.mkdir(parents=True, exist_ok=True)
(out / "weights.bin").write_bytes(b"".join(chunks))
json.dump({"dtype": "float32", "total": off, "tensors": manifest}, open(out / "manifest.json", "w"), indent=1)
n = np.load(norm_path)
json.dump({"mean": n["mean"].reshape(-1).tolist(), "std": n["std"].reshape(-1).tolist()}, open(out / "norm.json", "w"), indent=1)
print("exported", off, "values to", out)
