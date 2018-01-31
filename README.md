### For yolov2

#### Data preparation

```shell
cd data/Pascal_VOC
ln -s /your/path/to/VOCdevkit .
python get_list.py
# change related path in script 
./generate_lmdb.sh 

```



