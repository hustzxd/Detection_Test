### For yolov2
#### YOLO Inference
```shell
python yolov2_show.py
```

![example1](http://localhost:8080/AIGroup/Detection_Test/blob/master/data/Pascal_VOC/example_images/example1.png)

#### Data preparation

```shell
cd data/Pascal_VOC
ln -s /your/path/to/VOCdevkit .
python get_list.py
# change related path in script 
./generate_lmdb.sh 

```



