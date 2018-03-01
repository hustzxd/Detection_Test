#!/usr/bin/env sh
set -e

HOME=/home/zhaoxiandong
TOOLS=$HOME/projects/caffe/build/tools

# WEIGHTS=$HOME/data/bvlc_reference_caffenet.caffemodel
$TOOLS/caffe test \
    --model=$1 \
    --weights=$2 \
    --iterations=100 \
    --gpu=1
