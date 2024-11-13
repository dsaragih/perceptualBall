#!/bin/bash

# Check if an argument is provided
if [ -n "$1" ]; then
    IMAGE_ARG="--image $1"
else
    IMAGE_ARG=""
fi

# Just the name e.g. "results"
if [ -n "$2" ]; then
    RESULTS_ARG="--out_dir $2"
else
    RESULTS_ARG=""
fi

# Just the name e.g. "results"
if [ -n "$3" ]; then
    TARGET=$3
else
    TARGET="94"
fi

python run.py \
    --k 20 \
    --epochs 5 \
    --ga 1000 \
    --sa 0 \
    --ds 10 \
    --targetID $TARGET \
    --ts 50 \
    --ns 0.1 \
    $IMAGE_ARG \
    $RESULTS_ARG
