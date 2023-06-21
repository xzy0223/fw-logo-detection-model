#!/bin/bash

# 安装s5cmd，此工具传输数据效率更高
rm -rf s5cmd/
git clone https://github.com/peak/s5cmd && cd s5cmd
docker build -t s5cmd . && cd ..

# 安装xml到dict的转换lib
pip install xmltodict

# 安装kaggle命令行工具，用于方便从kaggle下载logo3k数据集，身份认证token在kaggle.json中
pip install kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d lyly99/logodet3k

# 将logo3k数据集解压
rm -rf raw_data/
mkdir -p raw_data/LogoDet-3K
unzip -q -d raw_data logodet3k.zip
rm -rf logodet3k.zip

# 下载FW提供的logo数据集，请将s3地址替换成自己的
# zip包解压到raw_data, zip包解压后的目录结构如下：
# image-data-4.28
#   ｜--brand_map.csv
#   ｜--labels.txt
#   ｜--labels
#        ｜--xxxx.txt
#   ｜--images
#        ｜--xxxx.jpg
docker run --rm -v $(pwd):/aws -v ~/.aws:/root/.aws s5cmd cp s3://logo-detection-data/image-data-4.28.zip /aws/
unzip image-data-4.28 -d raw_data
rm -rf image-data-4.28.zip

# 准备训练数据的目录结构，包括了基础的logo3k和FW5K
rm -rf train_data/
mkdir -p train_data/LogoDet-3K/datasets/images/train
mkdir -p train_data/LogoDet-3K/datasets/images/val
mkdir -p train_data/LogoDet-3K/datasets/labels/train
mkdir -p train_data/LogoDet-3K/datasets/labels/val
mkdir -p train_data/LogoDet-3K/cfg/
mkdir -p train_data/LogoDet-3K/weights/
mkdir -p train_data/FreeWheel-5K-by-video-name/datasets/images/train
mkdir -p train_data/FreeWheel-5K-by-video-name/datasets/images/val
mkdir -p train_data/FreeWheel-5K-by-video-name/datasets/labels/train
mkdir -p train_data/FreeWheel-5K-by-video-name/datasets/labels/val
mkdir -p train_data/FreeWheel-5K-by-video-name/cfg/
mkdir -p train_data/FreeWheel-5K-by-video-name/weights/
#wget -P train_data/LogoDet-3K/weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
#cp train_data/LogoDet-3K/weights/yolov8s.pt train_data/FreeWheel-5K-by-video-name/weights/
#chmod 600 train_data/LogoDet-3K/weights/yolov8s.pt

# 将FW5k的数据集复制到要用于训练的目录
cp raw_data/image-data-4.28/images/*.jpg train_data/FreeWheel-5K-by-video-name/datasets/images/train
cp raw_data/image-data-4.28/labels/*.txt train_data/FreeWheel-5K-by-video-name/datasets/labels/train
