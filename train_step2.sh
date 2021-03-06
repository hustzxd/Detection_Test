#!/usr/bin/env sh
set -e

HOME=/home/zhaoxiandong
TOOLS=$HOME/projects/caffe/build/tools
cur_date=`date +%Y-%m-%d-%H-%M-%S`
cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
log_file_name="$cur_dir/log/step2-${cur_date}.log"

solver=models/darknet_yolov2/solver-voc.2.0_step2.prototxt
# WEIGHTS=$HOME/data/bvlc_reference_caffenet.caffemodel
$TOOLS/caffe train \
    --solver=$solver \
    --weights=$1 \
    --gpu=0 2>&1 | tee -a ${log_file_name}
