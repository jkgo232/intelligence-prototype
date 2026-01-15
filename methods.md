\# Methods



\## Overview



The goal is to build a toy-model that takes raw telescope data and converts it into reproducible, multi-modal datasets and analyzes these using a combination of classical signal processing, machine learning, and large language models (LLMs).

---



\## Data Sources



The data source used for the development of this project is publicly available radio telescope data from the \*\*Breakthrough Listen Open Data Archive\*\*, collected with the Green Bank Telescope (GBT).



Raw data files are not included in this repository due to size constraints. Instructions for obtaining the data are provided below.



\### Data Access



Example datasets may be downloaded from the Breakthrough Listen archive: https://breakthroughinitiatives.org/opendatasearch



After download, raw HDF5 files should be placed in: data/raw/



---



\## Preprocessing and RFI Mitigation



Raw spectrogram data are preprocessed to reduce instrumental artifacts and common sources of terrestrial radio-frequency interference (RFI).



Preprocessing steps include:



1\. \*\*Data loading\*\*

&nbsp;  -  HDF5 files are loaded using the `blimpy` library.

&nbsp;  -  Spectrograms are extracted as 2D time–frequency arrays.



2\. \*\*Baseline subtraction\*\*

&nbsp;  -  A median value is subtracted along the time axis to suppress slowly varying backgrounds.



3\. \*\*Normalization\*\*

&nbsp;  -  Spectrograms are normalized using robust statistics (median and standard deviation) to facilitate downstream ML training.



4\. \*\*RFI masking\*\*

&nbsp;  -  Frequency channels associated with known RFI bands are masked. (This is currently done minimally and needs to be edited by hand)



The result of this stage is a cleaned, normalized spectrogram suitable for tiling and feature extraction.



---



\## Spectrogram Tiling and Dataset Construction



To create CNN inputs, preprocessed spectrograms are divided into fixed-size tiles.



* &nbsp;Tile size: 128 × 128 (time × frequency)
* &nbsp;Overlap: default 50% - can be changed
* &nbsp;Tiles with negligible variance or power are discarded



Each tile is saved as a NumPy array and accompanied by metadata describing its location in time and frequency space.

This tiling process produces a dataset of spectrogram "images" suitable for convolutional neural networks (CNNs) and anomaly detection models.



---



\## Feature Extraction



In addition to image-based representations, a set of signal features is saved for each spectrogram tile. 



Extracted features include:

* Mean normalized power
* Standard deviation 
* Maximum normalized power 
* Kurtosis 
* Spectral entropy

All extracted features are stored in tabular format and linked to their corresponding spectrogram tiles.



---



\## Machine Learning Models



\### CNN Embedding Model



A small convolutional neural network is trained to map spectrogram tiles into a low-dimensional embedding space.

These embeddings are used for visualization, clustering, and anomaly detection.



---



\### Anomaly Detection



Given the limited availability of labeled technosignature examples, anomaly detection is performed using unsupervised methods.



Implemented approaches include:

* &nbsp;Autoencoder reconstruction error
* &nbsp;Distance-based outlier detection in embedding space
* &nbsp;Feature-based anomaly scoring



The outputs of these methods are combined into a single anomaly score used to rank candidate signals for further inspection.



---



\## Synthetic Signal Injection



To evaluate the model sensitivity and validate the pipeline, synthetic signals are injected into a small fraction background data tiles.



Injected signals vary in:

* &nbsp;Signal-to-noise ratio
* &nbsp;Frequency drift rate
* &nbsp;Duration
* &nbsp;Bandwidth



Recovery efficiency as a function of signal parameters is used as a basic performance metric.



---



\## Large Language Models as Reasoning Layers



Large Language Models (LLMs) are evaluated as \*\*interpretive reasoning layers\*\*, not as primary classifiers.



LLMs operate on structured textual summaries of signal properties derived from the feature extraction stage. For each candidate signal, an LLM is prompted to:



* &nbsp;Assess consistency with known RFI or astrophysical phenomena
* &nbsp;Evaluate alignment with common technosignature heuristics
* &nbsp;Provide an explanation of its reasoning



Both commercial and open-source LLMs are evaluated for:

* &nbsp;Consistency
* &nbsp;Agreement with heuristic expectations
* &nbsp;Completeness of explanations



---



\## Evaluation and Limitations



This is, again, a rough toy-model that is in dire need of optimization at every step. 



---



\## Future Work



In addition to optimizing the current pipeline, other extensions include:

* &nbsp;Compatibility with more LLMs
* &nbsp;LLM Fine-Tuning
* &nbsp;Application to interferometric datasets (e.g., ATA, VLA, MeerKAT)
* &nbsp;Real-time inference optimization (e.g., TensorRT)



---









