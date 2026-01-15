import sys 
import os 
from preprocess import preprocess 
from tile_waterfall import tile 

'''
Usage: python get_tiles.py '../data/raw/your_data_file.h5'

turns raw data into pre-processed tiles ready for scoring by CNN 
'''

def main(data_file):
    print(f"Processing file: {data_file}")
    base = os.path.splitext(os.path.basename(data_file))[0] 
    tiles_dir = f"../data/tiles/{base}" 
    features_path = f"../data/processed/{base}_features.npy"
    # Step 1: preprocess - saves raw and processed spectrum and waterfall - see preprocess.py to adjust downsizing and RFI masking
    spec, freqs = preprocess(data_file) 
    print("Preprocessing complete:", spec.shape)
    # Step 2: tiling currently in 128x128 tiles - adjust settings in tile_waterfall.py to change tile and stride size
    n_tiles = tile(spec, freqs) 
    print(f"Tiling complete: {n_tiles} tiles")


if __name__ == "__main__": 
     if len(sys.argv) != 2: 
           print("Usage: python main.py <data_file.h5>") 
           sys.exit(1)
     main(sys.argv[1])
