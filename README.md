# 人脸表情识别与修改
该目录包含了一个人脸表情识别与修改的项目代码。该项目基于Pytorch实现，其中人脸表情识别部分使用了[ResNet-18](https://pytorch.org/vision/stable/generated/torchvision.models.resnet18.html)的网络结构，人脸表情修改部分则主要参考[StarGAN](https://github.com/yunjey/StarGAN)进行实现。

## 依赖库
pytorch 1.1.0

torchvision 0.3.0

imageio 2.8.0

scikit-image 0.15.0

## 数据下载
实验数据集使用了[Fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)和[Expw](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)数据集。下载后需要使用`./src/scripts`中的脚本对数据集进行处理。

经过处理的人脸表情图像数据集和训练完成的模型文件已经上传至[百度网盘](https://pan.baidu.com/s/16BFCenZMw82rvTKtHf_egw)(提取码：qwer)，

## 训练
训练表情识别网络请运行`train-recognition-net.py`,训练表情修改网络请运行`train-synthesisnet.py`

## 测试
测试表情识别网络请运行`test_classification.py`，测试表情修改网络请运行`generate_sample_images.py`