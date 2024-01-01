# FreMIM（WACV 2024）

This repo is the official implementation for: [FreMIM: Fourier Transform Meets Masked Image Modeling for Medical Image Segmentation](https://arxiv.org/abs/2304.10864).

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel
- fvcore
- setproctitle
- tensorboardX
- pickle

## Data Acquisition
- The multimodal brain tumor datasets (**BraTS 2019** & **BraTS 2020**) could be acquired from [here](https://ipp.cbica.upenn.edu/).

- The liver tumor dataset **LiTS 2017** could be acquired from [here](https://competitions.codalab.org/competitions/17094#participate-get-data).

## Data Preprocess (BraTS 2019 & BraTS 2020)
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

`python3 ./data/preprocess.py`

## Training
Run the training script on BraTS dataset. Distributed training is available for training the proposed TransBTS.

`sh train.sh`

## Testing 
If  you want to test the model which has been trained on the BraTS dataset, run the testing script as following.

`python3 test.py`

After the testing process stops, you can upload the submission file to [here](https://ipp.cbica.upenn.edu/) for the final Dice_scores.

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
