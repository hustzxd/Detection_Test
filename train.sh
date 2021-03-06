#!/usr/bin/env sh
set -e

HOME=/home/zhaoxiandong
TOOLS=$HOME/projects/caffe/build/tools

# WEIGHTS=$HOME/data/bvlc_reference_caffenet.caffemodel
GLOG_logtostderr=0 GLOG_log_dir=log/ \
$TOOLS/caffe train \
    --solver=$1 \
    --weights=$2 \
    --gpu=1