# SETI Intelligence Prototype

This repository contains a research-engineering prototype for technosignature search in radio astronomy data.

The project demonstrates an end-to-end pipeline that:
- Ingests open radio telescope data
- Builds multi-modal datasets (spectrograms, features, text)
- Applies anomaly detection models to identify candidate signals
- Evaluates open-source and commercial LLMs as reasoning layers for technosignature assessment

## Data

This project uses publicly available radio astronomy data from the Breakthrough Listen initiative.
Raw data are not included in this repository.

Instructions for downloading the data are provided in `methods.md`.

## Scope and Limitations

This is a prototype system intended for research and evaluation purposes only.
It does not claim detection of extraterrestrial signals.

## Repository Structure

src/ # Core pipeline code
notebooks/ # Exploration and evaluation
scripts/ # CLI and batch scripts
data/ # Ignored; see methods.md
