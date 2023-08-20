#!/bin/bash

cd ..

DATA=/home/ljw/datasets

DATASET=$1
NLAB=$2 # total number of labels
SETUP=$3
CUDA=$4

if [ ${DATASET} == ssdg_pacs ]; then
    # NLAB: 210 or 105
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == ssdg_officehome ]; then
    # NLAB: 1950 or 975
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == ssdg_digitsdg ]; then
    # NLAB: 300 or 150
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
fi

TRAINER=FixMatch
NET=cnn_digitsdg

for SEED in $(seq 1 10)
do
    
    if [ ${SETUP} == 1 ]; then
        S1=${D2}
        S2=${D3}
        S3=${D4}
        T=${D1}
    elif [ ${SETUP} == 2 ]; then
        S1=${D1}
        S2=${D3}
        S3=${D4}
        T=${D2}
    elif [ ${SETUP} == 3 ]; then
        S1=${D1}
        S2=${D2}
        S3=${D4}
        T=${D3}
    elif [ ${SETUP} == 4 ]; then
        S1=${D1}
        S2=${D2}
        S3=${D3}
        T=${D4}
    fi

    CUDA_VISIBLE_DEVICES=${CUDA} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --source-domains ${S1} ${S2} ${S3} \
    --target-domains ${T} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/fixmatch/${DATASET}.yaml \
    --output-dir /home/ljw/DA_DG/ssdg-benchmark/output_work4/${DATASET}/nlab_${NLAB}/${TRAINER}/${NET}/${T}/seed${SEED} \
    MODEL.BACKBONE.NAME ${NET} \
    DATASET.NUM_LABELED ${NLAB}
    
done