# Flight Data & Remote Sensing Datasets

This directory contains satellite remote sensing datasets and aerial archives used for training the Edge TinyML landing zone classifier.

## Assets

* **EuroSAT_RGB.zip**: Archive (89.9 MB) of Sentinel-2 satellite imagery (27,000 images, 64x64 RGB across 10 land cover classes). Not tracked in git; `python ml/download_dataset.py` downloads it (TLS-verified, SHA-256 checked) and extracts it.
* **eurosat/**: Extracted image directory (gitignored to prevent repository bloat). Automatically extracted during workstation setup or via python ml/download_dataset.py.
