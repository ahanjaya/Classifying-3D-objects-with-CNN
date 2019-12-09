# Classifying 3D objects of (ModelNet) with 3D CNN

The main idea of this guide is to apply several deep learning model on 3D CAD models for objects classifications.


## Getting Started

These instructions will show you how to train and run the deep learning model.



## Prerequisites

What things you need to run this codes:

1. Python3
2. CUDA       --> [CUDA-10.1 on Windows 10](/VFKfNv89SMy55DX4LCTm7Q)

Python installation package:
```
sudo pip3 install numpy
sudo pip3 install pptk
sudo pip3 install matplotlib
sudo pip3 install itertools
sudo pip3 install scikit-learn
sudo pip3 install keras
sudo pip3 install tensorflow-gpu
```

After install tensorflow-gpu --> [Tensorflow on CUDA-10.1 & CUDA-10.2](/goG6L1h8T3i8UWmaM9MOtw)

Tested on:
1. 8th Gen. Intel® Core™ i7 processor
2. GTX 1080Ti 11GB GDDR5
3. Memory 32GB DDR4 2666MHz
4. SSD 256 GB
5. Ubuntu 16.04.06 LTS (with ROS Kinetic)


## Download Dataset
```
mkdir data
cd data/
wget http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
```


## Voxelization
1. Install binvox library
```
wget http://www.patrickmin.com/binvox/linux64/binvox?rnd=1520896952313989 -O binvox
chmod 755 binvox
wget http://www.patrickmin.com/viewvox/linux64/viewvox?rnd=1575610255146400 -O viewvox
chmod 755 viewvox
```

2. Convert all *.off files to *.binvox
```
python3 binvox_converter.py data/ModelNet10/ --remove-all-dupes
python3 binvox_converter.py data/ModelNet40/ --remove-all-dupes
```

3. Visualize .binvox data

`./viewvox <filename>.binvox`

## Training

`python3 deep_learning.py`

## Display Data

`python3 display_data.py`
