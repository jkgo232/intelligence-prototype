import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import tensorflow as tf 
from tensorflow.keras import layers, models 
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, entropy


'''
Loads tiles, validates data, injects artificial signal, trains, provides diagnostic statistics, and saves a simple CNN. 

1. loads tiles and validates size and finite-values
2. checks for duplicate tiles and prints label fractions
3. creates tiles with positive detections (simulated drift signal) for training -- adjust signal settings and fraction as needed
4. assignes injected signals with label=1 and plots them for visual inspection
5. measures mean, std, max, kurtosis, and entropy for each tile and saves as metadata - this can be plotted using check_features.py
6. checks global max and min and saves a plot of the value distribution
7. splits data into training, testing, and validating sets -- adjust fractions as needed
8. define CNN - adjust as needed NOTE: need to change shape if tile size changes!
9. trains and saves CNN
10. plots and prints diagnostics
11. ranks scores, plots top ten with scores and labels for evaluation

'''


#define functions:

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

def normalize_per_tile(X): 
    mu = X.mean(axis=(1,2), keepdims=True) 
    sigma = X.std(axis=(1,2), keepdims=True) + 1e-6 
    return (X - mu) / sigma        #preserves injected contrast



def train_CNN():

 #config

 TILES_DIR = "../data/tiles" 
 FIGURES_DIR = "../figures" 
 MODEL_DIR = "../models/" 
 OUTPUT_PATH = "../data/features.npy"

 TILE_SHAPE = (128, 128) 
 EPOCHS = 15 
 BATCH_SIZE = 32 
 RANDOM_SEED = 42 

 os.makedirs(FIGURES_DIR, exist_ok=True) 
 os.makedirs(MODEL_DIR, exist_ok=True) 
 np.random.seed(RANDOM_SEED) 
 tf.random.set_seed(RANDOM_SEED)

 # load tiles and inject an artificial signal

 tiles = [] 
 
 for fname in tqdm(os.listdir(TILES_DIR)): 
    if not fname.endswith(".npy") or fname == "metadata.npy": 
         continue 
    tile = np.load(os.path.join(TILES_DIR, fname))
    if tile.shape != TILE_SHAPE: 
         continue
    if not np.isfinite(tile).all(): 
         continue 

    tiles.append(tile)

 tiles = np.array(tiles) 

 import hashlib 
 def hash_tile(tile): 
    return hashlib.md5(tile.tobytes()).hexdigest()
 hashes = [hash_tile(t) for t in tiles] 
 print("Unique tiles:", len(set(hashes)), "/", len(hashes))

 print(f"Loaded {len(tiles)} validated tiles") 

 INJECTION_FRACTION = 0.3
 rng = np.random.default_rng(42)
 tiles_injected = []
 labels = []
 features = []
 i=0

 for tile in tiles:
    if rng.random() < INJECTION_FRACTION:
       plt.figure(figsize=(10, 4))
       plt.subplot(1, 2, 1)
       plt.title("Original")
       plt.imshow(tile, aspect="auto", origin="lower")
       plt.colorbar()
       tile = inject_drifting_tone(tile, snr=rng.uniform(6, 12), drift_rate=rng.uniform(-0.5, 0.5), width=rng.integers(1, 3), rng=rng,)
       plt.subplot(1, 2, 2)
       plt.title("Injected")
       plt.imshow(tile, aspect="auto", origin="lower")
       plt.colorbar()
       plt.savefig(os.path.join(FIGURES_DIR, "tile_injected_signal_test"+str(i)+".png"))
       plt.close()
       i+=1
       label = 1
    else:
       label = 0
    labels.append(label)
    tiles_injected.append(tile)

    # Flatten for statistics
    flat = tile.flatten()
    feat = { "mean": np.mean(flat), "std": np.std(flat), "max": np.max(flat), "kurtosis": kurtosis(flat), "entropy": entropy(np.histogram(flat, bins=50)[0] + 1),}
    features.append(feat)

 tiles_injected = np.array(tiles_injected)
 labels = np.array(labels)
 np.save(OUTPUT_PATH, features)
 print(f"Saved features for {len(features)} tiles")

 tiles=tiles_injected
 print("Label counts:", np.unique(labels, return_counts=True))

 # data validation

 all_values = tiles.flatten()
 print("Global min:", all_values.min())
 print("Global max:", all_values.max())
 plt.hist(all_values, bins=100)
 plt.xlabel("Normalized Power ( ^c)")
 plt.ylabel("Count")
 plt.yscale('log')
 plt.title("Global Tile Value Distribution")
 plt.tight_layout()
 plt.savefig(os.path.join(FIGURES_DIR, "tile_value_histogram.png"), dpi=150)
 plt.close()

 # split data

 X_train, X_temp, y_train, y_temp = train_test_split( tiles, labels, test_size=0.3, random_state=RANDOM_SEED, stratify=labels )
 X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp )
 print("Train:", len(X_train))
 print("Val: ", len(X_val))
 print("Test: ", len(X_test))

# normalize per tile

 X_train = normalize_per_tile(X_train) 
 X_val = normalize_per_tile(X_val) 
 X_test = normalize_per_tile(X_test)
 
 # Add channel dimension
 X_train = X_train[..., np.newaxis]
 X_val = X_val[..., np.newaxis]
 X_test = X_test[..., np.newaxis]

 # define CNN

 model = models.Sequential([ layers.Input(shape=(128, 128, 1)), layers.Conv2D(16, 3, activation="relu", padding="same"), 
    layers.Conv2D(32, 3, activation="relu", padding="same"), layers.Conv2D(64, 3, activation="relu", padding="same"), 
    layers.GlobalAveragePooling2D(), layers.Dense(1, activation="sigmoid"),]) 
 model.compile( optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")], ) 
 model.summary()

 callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
 
 # train and save

 history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, )

 model.save(os.path.join(MODEL_DIR, "cnn_baseline_inj.keras"))
 print("Model saved.")

 # plot and print diagnostics

 plt.plot(history.history["loss"], label="train") 
 plt.plot(history.history["val_loss"], label="val") 
 plt.xlabel("Epoch") 
 plt.ylabel("Loss") 
 plt.legend() 
 plt.title("Training Loss") 
 plt.tight_layout() 
 plt.savefig(os.path.join(FIGURES_DIR, "training_loss.png"), dpi=150) 
 plt.close()

 test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0) 
 print(f"Test accuracy: {test_acc:.3f}") 
 print(f"Test AUC: {test_auc:.3f}")

 scores = model.predict(X_test).squeeze() 
 top_idx = np.argsort(scores)[-10:] 
 for i in top_idx: 
  plt.imshow(X_test[i].squeeze(), aspect="auto", origin="lower") 
  plt.title(f"Score={scores[i]:.2f}, Label={y_test[i]}") 
  plt.colorbar(label="Normalized Power (σ)") 
  plt.tight_layout() 
  plt.savefig(os.path.join(FIGURES_DIR, "tile_"+str(i)+".png"), dpi=150) 
  plt.close()

 return 0

train_CNN()





