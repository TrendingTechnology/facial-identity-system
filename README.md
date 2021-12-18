# Facial Identity system

## ⭐️⭐️⭐ This repo is still updating ⭐⭐⭐

## Introduction

This project is to utilize facial recognition to create a facial identity system. Our backend is constructed by one-shot models which is more flexible for adding a new face. The system is built on personal computer and Jetson Nano. Jetson Nano is used to recognized the faces and upload the detected information to Firebase. Users who used our application with account and password can log in to control the database and also see the information.

### Folder structure

```
| - backend - For Personal computer
|
| - csv_file - Contribution for the CelebA dataset
|
| - jetson - Files for Jetson Nano
|
| - model - Model we used for training and detecting
```

### Features

Our facial identity system includes below features:

* One-shot face recognition, add your faces without extra training
* Complete database operation (upload, delete, update)
* Fine-tuned your model at any time
* Use as a monitor
* Visualize the features

## Installation

### Personal computer

```shell
$ pip install -r requirements.txt
```

### Jetson Nano

```shell
$ pip install -r requirements.txt
```

### Increase swap space on Jetson Nano (Optional)

> Our nano would crush when using cuda until we increase its swap memory 🥳

```shell
# 4.0G is the swap space
$ sudo fallocate -l 4.0G /swapfile
$ sudo chmod 600 /swapfile
$ sudo mkswap /swapfile
$ sudo swapon /swapfile

# Create swap memory on every reboot
$ sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
```

## Experiments
### Result for real-time training

<table>
    <tr>
        <td colspan="2">Type</td>
        <td>Original</td>
        <td>New</td>
    </tr>
    <tr>
        <td rowspan="2">Cosine Similarity</td>
        <td>Positive</td>
        <td>0.9889</td>
        <td>0.9863</td>
    </tr>
    <tr>
        <td>Negative</td>
        <td>0.7673</td>
        <td>0.6695</td>
    </tr>
    <tr>
        <td rowspan="2">L2 Distance</td>
        <td>Positive</td>
        <td>0.1491</td>
        <td>0.1655</td>
    </tr>
    <tr>
        <td>Negative</td>
        <td>0.6822</td>
        <td>0.8130</td>
    </tr>
</table>

### Run time using different methods

* **second per image (s / img)**

| CPU (Pytorch) | Cuda (Pytorch) |  ONNX   | TensorRT |
|:-------------:|:--------------:|:-------:|:--------:|
|     4.11s     |    75.329s     | 0.1260s |  1.975s  |

> It is surprising that cuda consumes lots of time. We guess it is because cuda rely on huge amount of swap memory that slow down its runtime 😢.

## Contribution to CelebA

In order to train one-shot model, we obtain the face's coordinates beforehand. All files are placed in `csv_file`.
> The coordinates were obtained from [facenet-pytorch](https://github.com/timesler/facenet-pytorch)

| File name         | Description                                                                       |
|-------------------|-----------------------------------------------------------------------------------|
| `id_multiple.csv` | To ensure each celebrity have at least two images (For positive usage).           |
| `cropped.csv`     | Include the face's coordinates and ensure each celebrity has at least two images. |

## Citation

```bib
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}

@inproceedings{koch2015siamese,
  title={Siamese neural networks for one-shot image recognition},
  author={Koch, Gregory and Zemel, Richard and Salakhutdinov, Ruslan and others},
  booktitle={ICML deep learning workshop},
  volume={2},
  year={2015},
  organization={Lille}
}

@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International conference on machine learning},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}

@inproceedings{schroff2015facenet,
  title={Facenet: A unified embedding for face recognition and clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={815--823},
  year={2015}
}
```
