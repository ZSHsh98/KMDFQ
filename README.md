# Generative Data Free Model Quantization with Knowledge Matching

We provide PyTorch implementation for "Generative Data Free Model Quantization with Knowledge Matching".  

<!-- ## Paper
* [Generative Low bitwidth Data Free Quantization](https://arxiv.org/abs/2003.03603)  -->


<br/>

## Dependencies

* Python 3.7
* PyTorch 1.7.1
* dependencies in requirements.txt

<br/>

## Getting Started

### Installation

1. Clone this repo:

        git clone https://github.com/zhangshuhai-shuhai/KMDFQ.git
        cd KMDFQ

2. Install pytorch and other dependencies.

        pip install -r requirements.txt

### Set the paths of datasets for testing
1. Set the "dataPath" in "cifar100_resnet20.hocon" as the path root of your CIFAR-100 dataset. For example:

        dataPath = "/home/datasets/Datasets/cifar"

2. Set the "dataPath" in "imagenet_resnet18.hocon" as the path root of your ImageNet dataset. For example:

        dataPath = "/home/datasets/Datasets/imagenet"

### Training

To quantize the pretrained ResNet-20 on CIFAR-10 to 4-bit:

    python main_D.py 
    --conf_path ./cifar10_resnet20.hocon \
    --id 1 \
    --CE_WEIGHT 1 \
    --BNS_WEIGHT 1 \ 
    --FEATURE_WEIGHT 1  
    --warmup_epochs 20 \ 
    --visible_devices 0 \
    --qw 4 \
    --qa 4

To quantize the pretrained ResNet-20 on CIFAR-100 to 5-bit:

    python main_D.py 
    --conf_path ./cifar100_resnet20.hocon \
    --id 1 \
    --CE_WEIGHT 1 \
    --BNS_WEIGHT 1 \ 
    --FEATURE_WEIGHT 1  
    --warmup_epochs 20 \ 
    --visible_devices 0 \
    --qw 5 \
    --qa 5

To quantize the pretrained ResNet-18 on ImageNet to 6-bit:

    python main_D.py 
    --conf_path ./imagenet_resnet18.hocon \
    --id 1 \
    --CE_WEIGHT 1 \
    --BNS_WEIGHT 1  \
    --FEATURE_WEIGHT 1 \
    --warmup_epochs 50 \
    --visible_devices 6 \
    --selenet resnet18 \
    --qw 6 \
    --qa 6


<br/>

## Results

|  Dataset | Model | Pretrain Top1 Acc(%) | W8A8| W6A6| W5A5| W4A4|
   | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
  | CIFAR-10 | ResNet-20| 93.89|94.05| 93.94 |93.67 |92.24|
  | CIFAR-100 | ResNet-20| 70.33 | 70.53| 70.35 |69.68 |67.15|
  | ImageNet | ResNet-18 | 71.58 |70.73 |70.61 |69.93| 64.39|
  | ImageNet | BN-VGG16 |74.38|72.33 |72.29 |71.89 |68.79|
  | ImageNet |Inception v3|77.63|76.45 |76.43 |75.54 |71.22|
  | ImageNet |MobileNet v2|73.08|72.58 |72.21| 71.03 |63.48|
  | ImageNet |ShuffleNet|65.16|61.90 |60.95 |56.73| 28.26|
  | ImageNet | ResNet-50 | 77.76|77.50|77.21|75.67|68.84

Note that we use the pretrained models from [pytorchcv](https://www.cnpython.com/pypi/pytorchcv).

<br/>

<!-- ## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/2003.03603):

    @InProceedings{xu2020generative,
    title = {Generative Low-bitwidth Data Free Quantization},
    author = {Shoukai, Xu and Haokun, Li and Bohan, Zhuang and Jing, Liu and Jiezhang, Cao and Chuangrun, Liang and Mingkui, Tan},
    booktitle = {The European Conference on Computer Vision},
    year = {2020}
    }

<br/>

## Acknowledgments
This work was partially supported by the Key-Area Research and Development Program of Guangdong Province 2018B010107001, Program for Guangdong Introducing Innovative and Entrepreneurial Teams 2017ZT07X183, Fundamental Research Funds for the Central Universities D2191240. -->