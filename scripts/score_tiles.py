import os 
import json 
import numpy as np 
import tensorflow as tf 
from tqdm import tqdm 
from scipy.stats import kurtosis, entropy
'''
Uses trained CNN model to score tiles and outputs into .json files ready for llM analysis:

1. loads trained CNN 
1. loads and validates tiles
2. injects a signal into a small fraction of tiles for validation of CNN and LLM analysis
4. normalizes tiles
5. scores tiles
5. measures mean, std, max, kurtosis, and entropy for each tile and saves as metadata - this can be plotted using check_features.py
6. prints tile numbers with injected signals and their CNN score for quick evaluation
7. ranks tiles and saves tile number, score, mean, std, max, kurtosis, and entropy of top 50 to scored_tiles.json
8. same as above but also saves injection = True/False and injection params to scored_tiles_inj.json for CNN/LLM performance validation
'''

# config
MODEL_PATH = "../models/cnn_baseline_inj.keras" 
TILES_DIR = "../data/tiles" # new, unseen tiles 
OUTPUT_JSON = "../outputs/scored_tiles.json" 
OUTPUT_JSON_p = "../outputs/scored_tiles_inj.json" 
TOP_K = 50 # how many top tiles to send to LLM 
TILE_SHAPE = (128, 128) 
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# define functions
def normalize_per_tile(tile, eps=1e-6): 
    mu = tile.mean() 
    sigma = tile.std() 
    return (tile - mu) / (sigma + eps)

def extract_features(tile): 
    """ Simple scalar features for LLM reasoning """ 
    flat = tile.flatten() 
    hist = np.histogram(flat, bins=50)[0] + 1 
    return {
        "mean": float(np.mean(flat)), 
        "std": float(np.std(flat)), 
        "max": float(np.max(flat)), 
        "kurtosis": float(kurtosis(flat)), 
        "entropy": float(entropy(hist)),
    }

def inject_drifting_tone(tile, snr=10.0, drift_rate=0.2, width=1, rng=None,):
     """
     Inject a narrowband drifting signal into a tile.
     Parameters
     ----------
     tile : 2D numpy array
     snr : float
        Signal-to-noise ratio (relative to tile std)
     drift_rate : float
        Pixels per time step
     width : int
         Frequency width in pixels
     """
     if rng is None:
         rng = np.random.default_rng()
     injected = tile.copy()
     T, F = injected.shape
     t0 = rng.integers(0, T)
     f0 =  rng.integers(F // 4, 3 * F // 4)
     noise_std = np.std(tile)
     amplitude = snr * noise_std
     for t in range(T):
          f = int(f0 + drift_rate * (t - t0))
          if 0 <= f < F:
             injected[t, max(0, f - width):min(F, f + width + 1)] += amplitude
     return injected

# load CNN

print("Loading model...") 
model = tf.keras.models.load_model(MODEL_PATH) 
model.summary()

# load, validate, and inject signal

raw_tiles = []

for fname in tqdm(os.listdir(TILES_DIR)):
    if not fname.endswith(".npy") or fname == "metadata.npy":
         continue
    tile = np.load(os.path.join(TILES_DIR, fname))
    # --- Shape check ---
    if tile.shape != TILE_SHAPE:
         continue
    # --- Finite-value check ---
    if not np.isfinite(tile).all():
         continue

    raw_tiles.append(tile)

raw_tiles = np.array(raw_tiles)


INJECTION_FRACTION = 0.04
rng = np.random.default_rng(42)
injection_log = [] 
tiles=[]
for idx, tile in enumerate(raw_tiles): 
    injected = False 
    inj_params = None 
    if rng.random() < INJECTION_FRACTION:
        snr = rng.uniform(6, 12) 
        drift = rng.uniform(-0.5, 0.5) 
        width = rng.integers(1, 3) 
        tile = inject_drifting_tone(tile, snr=snr, drift_rate=drift, width=width, rng=rng ) 
        injected = True 
        inj_params = {
            "snr": float(snr), 
            "drift_rate": float(drift), 
            "width": int(width),
        }
    injection_log.append({ 
        "tile_index": idx, 
        "injected": injected, 
        "params": inj_params
         })
    tiles.append(tile) 


# normalize, score, and store results

results = [] 
private_results=[]

for idx, tile in enumerate(tqdm(tiles)):
    if tile.shape != TILE_SHAPE or not np.isfinite(tile).all():
        continue

    # Normalize and predict
    tile_norm = normalize_per_tile(tile)
    x = tile_norm[np.newaxis, ..., np.newaxis]
    score = float(model.predict(x, verbose=0)[0, 0])
    features = extract_features(tile_norm)

    # Record for LLM (no injection info)
    results.append({
        "tile_id": f"tile_{idx:06d}",
        "cnn_score": score,
        "features": features,
    })

    # Record private info (including injection)
    injection_info = injection_log[idx] if "injection_log" in globals() else {"injected": False, "params": None}
    private_results.append({
        "tile_id": f"tile_{idx:06d}",
        "cnn_score": score,
        "features": features,
        "injected": injection_info["injected"],
        "injection_params": injection_info["params"],
    })

print(f"Scored {len(results)} tiles")

if "injection_log" in globals():
    injected_scores = [(i, results[i]["cnn_score"]) for i, info in enumerate(injection_log) if info["injected"]]
    print("Injected tile scores:", injected_scores)

# rank tiles
results_sorted = sorted(results, key=lambda r: r["cnn_score"], reverse=True) 
top_results = results_sorted[:TOP_K]

private_results_sorted = sorted(private_results, key=lambda r: r["cnn_score"], reverse=True) 
top_private_results = private_results_sorted[:TOP_K]

# print results for llm
with open(OUTPUT_JSON, "w") as f: 
    json.dump(top_results, f, indent=2) 
print(f"Saved top {TOP_K} tiles to {OUTPUT_JSON}")
with open(OUTPUT_JSON_p, "w") as f: 
    json.dump(top_private_results, f, indent=2) 



