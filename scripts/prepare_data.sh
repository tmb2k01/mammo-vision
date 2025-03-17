#!/bin/bash

python -m scripts.prepare_data_detection --input-directory data/cbis-ddsm
python -m scripts.prepare_data_segmentation --input-folder data/cbis-ddsm-detec
