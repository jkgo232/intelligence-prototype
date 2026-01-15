import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import tensorflow as tf 
from tensorflow.keras import layers, models 
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, entropy

'''
Loads tiles, validates data, trains, provides diagnostics, and saves a simple CNN. NO INJECTED SIGNALS

1. loads tiles and validates size and finite-values
2. applies labels, currently anything with a signal higher than 3-sigma is labeled '1' all else are '0'
3. measures mean, std, max, kurtosis, and entropy for each tile and saves as metadata - this can be plotted using check_features.py
4. checks for duplicate tiles and prints label fractions 
5. checks global max and min and saves plot of value distribution
6. splits data into training, testing, and validating sets -- adjust fractions as needed 
7. define CNN - toy-model, adjust as needed NOTE: need to change shape if tile size changes!
8. trains and saves CNN
9. plots and prints diagnostics
10. ranks scores, plots top ten with scores and labels for evaluation

'''

def train_CNN():

 # Config

 TILES_DIR = "../data/tiles" 
 FIGURES_DIR = "../figures" 
 MODEL_DIR = "../models/" 
 OUTPUT_PATH = "../data/features.npy"

 TILE_SHAPE = (128, 128) 
 SIGNAL_THRESHOLD = 3.0 # weak-label threshold (σ units) 
 EPOCHS = 15 
 BATCH_SIZE = 32 
 RANDOM_SEED = 42 

 os.makedirs(FIGURES_DIR, exist_ok=True) 
 os.makedirs(MODEL_DIR, exist_ok=True) 
 np.random.seed(RANDOM_SEED) 
 tf.random.set_seed(RANDOM_SEED)

 # load tiles

 tiles = [] 
 labels = []
 features = []
 
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

    tiles.append(tile)

    # --- apply labels ---
    label = 1 if tile.max() > SIGNAL_THRESHOLD else 0 
    labels.append(label) 

    # Flatten for statistics
    flat = tile.flatten()
    feat = { "mean": np.mean(flat), "std": np.std(flat), "max": np.max(flat), "kurtosis": kurtosis(flat), "entropy": entropy(np.histogram(flat, bins=50)[0] + 1),}
    features.append(feat)

 np.save(OUTPUT_PATH, features)
 print(f"Saved features for {len(features)} tiles")

 tiles = np.array(tiles) 
 labels = np.array(labels) 

 import hashlib 
 def hash_tile(tile): 
    return hashlib.md5(tile.tobytes()).hexdigest()
 hashes = [hash_tile(t) for t in tiles] 
 print("Unique tiles:", len(set(hashes)), "/", len(hashes))


 print(f"Loaded {len(tiles)} validated tiles") 
 print(f"Signal-like fraction: {labels.mean():.3f}")

 # data validation

 all_values = tiles.flatten() 
 print("Global min:", all_values.min()) 
 print("Global max:", all_values.max()) 
 plt.hist(all_values, bins=100) 
 plt.xlabel("Normalized Power (σ)") 
 plt.ylabel("Count") 
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


 # normalize
 mean = X_train.mean() 
 std = X_train.std() 
 X_train = (X_train - mean) / std 
 X_val = (X_val - mean) / std 
 X_test = (X_test - mean) / std

 # Add channel dimension
 X_train = X_train[..., np.newaxis] 
 X_val = X_val[..., np.newaxis] 
 X_test = X_test[..., np.newaxis]

 # define CNN

 model = models.Sequential([ layers.Input(shape=(128, 128, 1)), layers.Conv2D(16, (3, 3), activation="relu", padding="same"), layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"), layers.MaxPooling2D((2, 2)), layers.Conv2D(64, (3, 3), activation="relu", padding="same"), 
    layers.GlobalAveragePooling2D(), layers.Dense(32, activation="relu"), layers.Dense(1, activation="sigmoid"),]) 
 model.compile( optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")], ) 
 model.summary()

 callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
 
# train and save model
 history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, )

 model.save(os.path.join(MODEL_DIR, "cnn_baseline.keras"))
 np.save("../models/train_mean.npy", mean) 
 np.save("../models/train_std.npy", std) 
 print("Model saved.")

 # plot diagnostics
 plt.plot(history.history["loss"], label="train") 
 plt.plot(history.history["val_loss"], label="val") 
 plt.xlabel("Epoch") 
 plt.ylabel("Loss") 
 plt.legend() 
 plt.title("Training Loss") 
 plt.tight_layout() 
 plt.savefig(os.path.join(FIGURES_DIR, "training_loss.png"), dpi=150) 
 plt.close()

 # print results
 test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0) 
 print(f"Test accuracy: {test_acc:.3f}") 
 print(f"Test AUC: {test_auc:.3f}")

 # rank tiles and plot to check performance
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
