import matplotlib.pyplot as plt
import numpy as np

#Plots random tiles and saves images for a quick sanity check

OUTPUT_PATH = "../figures/"
test=[101, 500, 676, 998]
for i in test:
 tile = np.load("../data/tiles/tile_000"+str(i)+".npy") 
 plt.imshow(tile, aspect="auto", origin="lower") 
 plt.colorbar(label="Normalized Power (σ)") 
 plt.savefig(OUTPUT_PATH+str(i)+".png")
 plt.close()
 print(f"Saved tile image to {OUTPUT_PATH}"+str(i)+".png")

