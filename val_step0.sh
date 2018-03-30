#!/usr/bin/env sh
set -e

HOME=/home/zhaoxiandong
TOOLS=$HOME/projects/caffe/build/tools
Model=models/darknet_yolov2/train_val-voc.2.0_step0.prototxt
# WEIGHTS=$HOME/data/bvlc_reference_caffenet.caffemodel
$TOOLS/caffe test \
    --model=$Model \
    --weights=$1 \
    --iterations=100 \
    --gpu=1
