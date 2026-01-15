from blimpy import Waterfall 
import matplotlib.pyplot as plt
import numpy as np

'''
This script: 
1. reads in raw data 
2. plots and saves raw waterfall and spectrum
3. prints resolution
4. downsamples along the time axis  
5. normalizes data and removes specified RFI channels
6. plots post-processed waterfall and spectrum

The steps here are specific to the data used which had high time resolution and low frequency resolution, downsampling will need
to be adjusted to the data set.  Can be altered to use fixed time and frequency resolution.

Set RFI bands for removal by hand
'''

def preprocess(data):

 # ------------------------------- Config -------------------------------
 DATA_PATH = str(data) 

 OUTPUT_PATH_rspec = "../figures/raw_spectrum.png"
 OUTPUT_PATH_rwf = "../figures/raw_waterfall.png"
 OUTPUT_PATH_spec = "../figures/processed_spectrum.png"
 OUTPUT_PATH_wf = "../figures/processed_waterfall.png"

 # Conservative, well-known RFI bands (MHz), can be edited as needed
 RFI_BANDS_MHZ = [
 (1805, 1880), # LTE Band 3 
 (2110, 2170), # LTE Band 1 
 (2620, 2690), # LTE Band 7 
 (2300, 2400), # LTE Band 40 
 (2496, 2690), # 5G 
 #(1170, 1180), # GPS L5 
 #(1210, 1230), # GPS L2 
 #(1570, 1580), # GPS L1 
 #(1610, 1630), # Iridium 
 ]
 # ------------------------------- Load data -------------------------------
 wf = Waterfall(DATA_PATH) 
 spec = wf.data.squeeze() # (time, frequency) 
 freqs_mhz = wf.get_freqs() 
 times = np.arange(spec.shape[0]) * wf.header["tsamp"]

 # ------------------------------- Check Resolution -------------------------------

 freqs = wf.get_freqs() # MHz 
 df_mhz = np.median(np.absolute(np.diff(freqs))) 
 print("Frequency resolution:", df_mhz, "MHz") 
 print("~", df_mhz * 1e6, "Hz")

 hdr = wf.header 
 for k in hdr: 
  print(k, ":", hdr[k])


 dt = hdr["tsamp"] # seconds 
 print("Time resolution:", dt, "seconds")
 print("Time shape:", spec.shape)


 total_time = spec.shape[0] * dt 
 print("Total time (s):", total_time) 

 dt_raw = hdr["tsamp"] # seconds 
 target_dt = 0.5 #downsize the time resolution  
 time_factor = int(target_dt / dt_raw)


 def downsample_time(spec, factor): 
  n_time = (spec.shape[0] // factor) * factor 
  spec_trim = spec[:n_time] 
  return spec_trim.reshape(-1, factor, spec.shape[1]).mean(axis=1)


 spec_ds = downsample_time(spec, time_factor)

 print("Downsampled shape:", spec_ds.shape)

 # ------------------------------- Plot Raw Data------------------------------- 
 raw_spectrum = np.median(spec, axis=0) 
 plt.figure(figsize=(10, 4)) 
 plt.plot(freqs_mhz, raw_spectrum, linewidth=0.8) 
 plt.xlabel("Frequency (MHz)") 
 plt.ylabel("Power (arbitrary units)") 
 plt.title("Raw Spectrum") 
 plt.grid(alpha=0.3) 
 plt.tight_layout() 
 plt.savefig(OUTPUT_PATH_rspec, dpi=150) 
 plt.close() 
 print(f"Saved raw spectrum to {OUTPUT_PATH_rspec}")

 plt.figure(figsize=(10, 6))
 im = plt.imshow(
     spec,
     aspect="auto",
     origin="lower",
     interpolation="nearest",
     extent=[
         freqs_mhz[0],
         freqs_mhz[-1],
         times[0],
         times[-1],
     ],
     cmap="viridis",
 )
 plt.xlabel("Frequency (MHz)")
 plt.ylabel("Time (s)")
 plt.title("Raw Waterfall")
 cbar = plt.colorbar(im)
 cbar.set_label("Power (arbitrary units)")
 plt.tight_layout()
 plt.savefig(OUTPUT_PATH_rwf, dpi=150)
 plt.close()
 print(f"Saved raw waterfall to {OUTPUT_PATH_rwf}")

 # ------------------------------- Preprocessing -------- 
 spec=spec_ds
 spec -= np.median(spec, axis=0)
 # Normalize
 spec /= np.std(spec)
 # Mask known RFI bands
 for fmin, fmax in RFI_BANDS_MHZ: 
   mask = (freqs_mhz >= fmin) & (freqs_mhz <= fmax) 
   spec[:, mask] = np.nan # use NaN so it disappears in plots

 spectrum = np.nanmean(spec, axis=0)

 # ------------------------------- Plot -------------------------------
 plt.figure(figsize=(10, 4)) 
 plt.plot(freqs_mhz, spectrum, linewidth=0.8) 
 plt.xlabel("Frequency (MHz)") 
 plt.ylabel("Normalized Power (σ units)") 
 plt.title("Time-Averaged Spectrum After Preprocessing") 
 plt.grid(alpha=0.3) 
 for fmin, fmax in RFI_BANDS_MHZ: 
    plt.axvspan(fmin, fmax, color="red", alpha=0.1)
 plt.tight_layout() 
 plt.savefig(OUTPUT_PATH_spec, dpi=150) 
 plt.close() 
 print(f"Saved processed spectrum to {OUTPUT_PATH_spec}")


 plt.figure(figsize=(10, 6))
 im = plt.imshow(
    spec,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    extent=[
        freqs_mhz[0],
        freqs_mhz[-1],
        times[0],
        times[-1],
    ],
    vmin=-3,
    vmax=3,
    cmap="viridis",
 )
 plt.xlabel("Frequency (MHz)")
 plt.ylabel("Time (s)")
 plt.title("Processed Waterfall (Baseline-Subtracted, Normalized)")
 cbar = plt.colorbar(im)
 cbar.set_label("Normalized Power (σ units)")
 plt.tight_layout()
 plt.savefig(OUTPUT_PATH_wf, dpi=150)
 plt.close()
 print(f"Saved processed waterfall to {OUTPUT_PATH_wf}")
 
 return spec, freqs_mhz
