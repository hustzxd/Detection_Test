#!/usr/bin/env sh

# CAFFE_ROOT=/home/i/Projects/caffe
CAFFE_ROOT=/home/zhaoxiandong/projects_i/caffe
#ROOT_DIR=/your/path/to/vocroot/
ROOT_DIR=./
LABEL_FILE=./label_map.txt

# mkdir $CAFFE_ROOT/data/yolov2/lmdb
# 2007 + 2012 trainval
LIST_FILE=./trainval.txt
LMDB_DIR=./train_lmdb
# rm -r $LMDB_DIR
PADDING=true
RESIZE_W=416
RESIZE_H=416
RGB=false   #future work

SHUFFLE=false #ture will be error>>> :(
$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE --add_padding=$PADDING \
  --encoded=true --encode_type=jpg --shuffle=$SHUFFLE \
  $ROOT_DIR $LIST_FILE $LMDB_DIR

# 2007 test
LIST_FILE=./test_2007.txt
LMDB_DIR=./test_lmdb
SHUFFLE=false

$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE --add_padding=$PADDING \
  --encoded=true --encode_type=jpg --shuffle=$SHUFFLE \
  $ROOT_DIR $LIST_FILE $LMDB_DIR 
