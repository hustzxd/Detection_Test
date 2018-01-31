## For yolov2
#### YOLO Inference
```shell
python yolov2_show.py
```

<img src="data/Pascal_VOC/example_images/example1.png" alt="Drawing" style="width: 200px;"/>
<img src="data/Pascal_VOC/example_images/example2.png" alt="Drawing" style="width: 200px;"/>
<img src="data/Pascal_VOC/example_images/example3.png" alt="Drawing" style="width: 200px;"/>

#### Data preparation

```shell
cd data/Pascal_VOC
ln -s /your/path/to/VOCdevkit .
python get_list.py
# change related path in script 
./generate_lmdb.sh 
```



