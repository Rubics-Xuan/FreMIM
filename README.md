# FreMIM（WACV 2024）

This repo is the official implementation for: 
[FreMIM: Fourier Transform Meets Masked Image Modeling for Medical Image Segmentation](https://arxiv.org/abs/2304.10864).

<h4 align="center"> <a href="https://rubics-xuan.github.io/FreMIM/" align="center"> [Project page] | </a> <a href="https://arxiv.org/abs/2304.10864" align="center"> [Paper] </h4> 
  
## Requirements
More details about our environmental requirements can be found in our uploaded "requirement.txt".
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel
- fvcore
- setproctitle
- tensorboardX
- pickle

## Data Acquisition & Data Preprocess
- Please refer to our previous open-sourced repo [TransBTS](https://github.com/Rubics-Xuan/TransBTS) for more specific details.

## Pre-Training with our FreMIM
- sh run_frequency.sh

## Fine-tuning & Testing
- Please refer to our previous open-sourced repo [TransBTS](https://github.com/Rubics-Xuan/TransBTS) for more specific details.

## Citation
If you use our code or model in your work or find it is helpful, please cite our paper:

```
@inproceedings{wang2024fremim,
  title={FreMIM: Fourier Transform Meets Masked Image Modeling for Medical Image Segmentation},
  author={Wang, Wenxuan and Wang, Jing and Chen, Chen and Jiao, Jianbo and Cai, Yuanxiu and Song, Shanshan and Li, Jiangyun},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7860--7870},
  year={2024}
}
```
