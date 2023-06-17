#!/bin/bash -i

DATA_PATH= # path/to/video/folder
METADATA_CSV_FILENAME= # path/to/metadata/csv/file
LOCAL_CHECKPOINT= # path/to/local/checkpoint/file.pth

BACKBONE=x3d # the backbone used in LOCAL_CHECKPOINT
STRIDE=1
DEVICE=cuda:0

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi

if [ -z "$METADATA_CSV_FILENAME" ]; then
    echo "METADATA_CSV_FILENAME variable is not set."
    echo "We provide metadata CSV files for ActivityNet and THUMOS14 in the data folder."
    exit 1
fi

if [ -z "$LOCAL_CHECKPOINT" ]; then
    echo "LOCAL_CHECKPOINT variable is not set."
    echo "Please set LOCAL_CHECKPOINT to the location of the local checkpoint .pth file."
    echo "Make sure to set the correct BACKBONE variable as well."
    exit 1
fi

if [ -z "$BACKBONE" ]; then
    echo "BACKBONE variable is not set."
    exit 1
fi

OUTPUT_DIR=output/local_checkpoint_${BACKBONE}_features/stride_${STRIDE}/

source activate tsp
mkdir -p $OUTPUT_DIR
export PYTHONPATH= # path/to/project

python extract_features.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--local-checkpoint $LOCAL_CHECKPOINT \
--backbone $BACKBONE \
--stride $STRIDE \
--device $DEVICE \
--output-dir $OUTPUT_DIR
