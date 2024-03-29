# Segmentation of Tiny Intracranial Hemorrhage via Learning-to-Rank Local Feature Enhancement

Implementation for ISBI 2024 paper <strong>Segmentation of Tiny Intracranial Hemorrhage via Learning-to-Rank Local Feature Enhancement</strong>
by Shizhan Gong, [Yuan Zhong](https://yzrealm.com/), Yuqi Gong, Nga Yan Chan, Wenao Ma, Calvin Hoi-Kwan Mak, Jill Abrigo, and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).

## Sample Results
![Alt text](assets/result.png?raw=true "Title")

## Setup and Instruction

Our code is based on the framework of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). The training data preprocessing and training protocol is exactly the same as the original nnU-Net.

Specifically, our modified variant can be found in the [script](nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_rank.py).

For installing nnU-Net, please refer to the [instructions](nnUNet/documentation/installation_instructions.md).

As the original repo is continuing upgrading, which may make our modification fail. Please install the historical version we provided in our repo.
```sh
cd nnUNet
pip install -e .
```

For data preprocessing, please refer to the [instructions](nnUNet/documentation/dataset_format.md).

For training and predicting nnU-Net, please refer to the [instructions](nnUNet/documentation/how_to_use_nnunet.md).

To train with our learning-to-rank variant, change the command as 

```sh
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_rank
```

To predict with our learning-to-rank variant, change the command as 
```sh
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_fullres -tr nnUNetTrainer_rank
```

## Sample data

We use the data stored in `.nii.gz` format, two sample cases can be found in the [sample_data](sample_data).

## Pre-trained Checkpoint
We provide several pre-trained checkpoints trained on our dataset correponding to different folds. You can download the [checkpoint](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155187960_link_cuhk_edu_hk/EpBtXy91doVOrFPfqA5YUmABKg9y1O9A-lwVtqFwXEBA-g?e=5pDB79) here.

## Bibtex
If you find this work helpful, you can cite our paper as follows:
```
@INPROCEEDINGS{gong2024segmentation,
  author={Gong, Shizhan and Zhong, Yuan and Gong, Yuqi and Chan, Nga Yan and Ma, Wenao and Mak, Calvin Hoi-Kwan and Abrigo, Jill and Dou, Qi},
  booktitle={2024 IEEE 21th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Segmentation of Tiny Intracranial Hemorrhage via Learning-to-Rank Local Feature Enhancement}, 
  year={2024}
 }
```
## Acknowledgement
Our code is based on  [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>