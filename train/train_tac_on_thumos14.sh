#!/bin/bash -i

ROOT_DIR=

if [ -z "$ROOT_DIR" ]; then
    echo "ROOT_DIR variable is not set."
    echo "Please set ROOT_DIR to the location of the THUMOS14 videos."
    echo "The directory must contain two subdirectories: valid and test"
    exit 1
fi

TRAIN_SUBDIR=valid
VALID_SUBDIR=test
TRAIN_CSV_FILENAME=../data/thumos14/thumos14_valid_tsp_groundtruth.csv
VALID_CSV_FILENAME=../data/thumos14/thumos14_test_tsp_groundtruth.csv
LABEL_COLUMNS=action-label
LABEL_MAPPING_JSONS=../data/thumos14/thumos14_action_label_mapping.json
LOSS_ALPHAS=1.0

BACKBONE=x3d

BATCH_SIZE=8
BACKBONE_LR=0.00001
FC_LR=0.002

OUTPUT_DIR=output/${BACKBONE}-tac_on_thumos14/backbone_lr_${BACKBONE_LR}-fc_lr_${FC_LR}/

MY_MASTER_ADDR=127.0.0.1
MY_MASTER_PORT=$(shuf -i 30000-60000 -n 1)

source activate tsp
mkdir -p $OUTPUT_DIR
export OMP_NUM_THREADS=6
export PYTHONPATH=/home/ubuntu/PycharmProjects/tsp/

python train.py \
--root-dir $ROOT_DIR \
--train-subdir $TRAIN_SUBDIR \
--valid-subdir $VALID_SUBDIR \
--train-csv-filename $TRAIN_CSV_FILENAME \
--valid-csv-filename $VALID_CSV_FILENAME \
--label-mapping-jsons $LABEL_MAPPING_JSONS \
--label-columns $LABEL_COLUMNS \
--loss-alphas $LOSS_ALPHAS \
--backbone $BACKBONE \
--batch-size $BATCH_SIZE \
--backbone-lr $BACKBONE_LR \
--fc-lr $FC_LR \
--output-dir $OUTPUT_DIR \
