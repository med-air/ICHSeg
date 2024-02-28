# Segmentation of Tiny Intracranial Hemorrhage via Learning-to-Rank Local Feature Enhancement

Implementation for ISBI 2024 paper Segmentation of Tiny Intracranial Hemorrhage via Learning-to-Rank Local Feature Enhancement
by Shizhan Gong, [Yuan Zhong](https://yzrealm.com/), Yuqi Gong, Nga Yan Chan, Wenao Ma, Calvin Hoi-Kwan Mak, Jill Abrigo, and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).

# Sample Results
![Alt text](assets/result.png?raw=true "Title")

# Setup and Instruction

Our code is based on the framework of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). The training data preprocessing and training protocol is exactly the same as the original nnU-Net.

Specifically, our modified variant can be found here:  

For installing nnU-Net, please refer to the instructions.

For data preprocessing, please refer to the instructions.

For training and predicting nnU-Net, please refer to the instructions.

To train with our learning-to-rank variant, change the command as 

```sh
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_rank
```

To predict with our learning-to-rank variant, change the command as 
```sh
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_fullres -tr nnUNetTrainer_rank
```

# Sample data

We use the data stored in `.nii.gz` format, two sample cases can be found in the sample_data folder.

# Pre-trained Checkpoint
We provide several pre-trained checkpoint trained on our dataset correponds to different folds. You can download the [checkpoint]().

## Bibtex
If you find this work helpful, you can cite our paper as follows:
```
@article{gong2024segmentation,
  title={Segmentation of Tiny Intracranial Hemorrhage via Learning-to-Rank Local Feature Enhancement},
  author={Gong, Shizhan and Zhong, Yuan and Gong, Yuqi and Chan, Nga Yan and Ma, Wenao and Mak, Calvin Hoi-Kwan and Abrigo, Jill and Dou, Qi},
  journal={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2024}
}
```

## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>