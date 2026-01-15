import os 
import numpy as np 
from blimpy import Waterfall 
from tqdm import tqdm


'''
Here we read in preprocessed data and generate 128x128 pixel tiles with
50% overlap,  to be analyzed by a trained CNN

Adjust tile and stride size as needed based on processed data resolution and desired targets
'''

def tile(spec, freqs_mhz):
 # ------------------------------- Config--------------------------- 
 OUTPUT_DIR = "../data/tiles" 
 TILE_TIME = 128 # time bins 
 TILE_FREQ = 128 # frequency channels 
 STRIDE_TIME = 64 
 STRIDE_FREQ = 64
 os.makedirs(OUTPUT_DIR, exist_ok=True)
 # ------------------------------- Tiling -------------------------------
 tiles = [] 
 metadata = [] 
 tile_id = 0 
 n_time, n_freq = spec.shape 
 for t in tqdm(range(0, n_time - TILE_TIME, STRIDE_TIME)):
    for f in range(0, n_freq - TILE_FREQ, STRIDE_FREQ): 
        tile = spec[t:t + TILE_TIME, f:f + TILE_FREQ]
        # Skip tiles with too much missing data
        nan_fraction = np.isnan(tile).mean() 
        if nan_fraction > 0.3: 
            continue
        # Replace NaNs for ML compatibility
        tile = np.nan_to_num(tile) 
        tile_path = os.path.join(OUTPUT_DIR, f"tile_{tile_id:06d}.npy") 
        np.save(tile_path, tile) 
        metadata.append({
            "tile_id": tile_id, "time_start": t, "freq_start_mhz": 
            freqs_mhz[f], "nan_fraction": nan_fraction,
        })
        tile_id += 1
 # Save metadata
 np.save(os.path.join(OUTPUT_DIR, "metadata.npy"), metadata) 
 print(f"Saved {tile_id} tiles to {OUTPUT_DIR}")
 return tile_id
