# DeepAttnMISL_MEDIA
Core codes of Pytorch implementation of MEDIA'20 paper and an improved version of MICCAI 19.

 - [Whole Slide Images Based Cancer Survival Prediction using Attention Guided Deep
Multiple Instance Learning Networks](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301535), [Jiawen Yao](https://utayao.github.io/), Xinliang Zhu, Jitendra Jonnagaddala, Nicholas Hawkins and Junzhou Huang.
<strong>Medical Image Analysis</strong>, Volume 65, October 2020, 101789, https://doi.org/10.1016/j.media.2020.101789

- [Deep Multi-instance Learning for Survival Prediction from Whole Slide Images](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_55), MICCAI 2019
---

### Overview
![Pipeline](https://ars.els-cdn.com/content/image/1-s2.0-S1361841520301535-fx1_lrg.jpg)

### Data
Need to specify the path of data label and image features
Data labels are in csv format. Image features can be saved in npz format with clustering label, etc. More can be found in dataset definition.

For example, patient 10000 has 100 sampled patches from at least 1 WSI, should stored as 10000.npz with the following information:
1. 'vgg_features': vgg features of each patch, so 100*L
2. 'pid': 10000, 1
3. 'time': OS time, 1
4. 'status': OS status, 1
5. 'img_path': path of each patch, for example, './10000/1.jpg', 100*1
6. 'cluster_num': cluster id of each patch. 100*1

More can be found in dataset definition.

#### Input file format

A csv file in the following formate is needed:
patient_ID | Img_patch_path | Survival_time | Survival_status |
--- | --- | --- | --- |
10000 |	/10000/1.jpg |	1000 |	0

### Training

Our implementation consists in a [main.py](./main.py) file from which are imported the MIL dataloader definition [MIL_dataloader.py](./MIL_dataloader.py), the model architecture [DeepAttnMISL_model.py](./DeepAttnMISL_model.py) and some miscellaneous training utilities.

After specific your label and feature path, run:
```
python main.py
```
The average C-index across 5 folds will show in the end. 

---
### Acknowledgments

- We thank for the data collection from NCI and UNSW. 
- This code is inspired by [SALMON](https://github.com/huangzhii/SALMON) and [AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL). 

---


### Citation
If you find this repository useful in your research, please cite:
```
@article{yao2020whole,
  title={Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks},
  author={Yao, Jiawen and Zhu, Xinliang and Jonnagaddala, Jitendra and Hawkins, Nicholas and Huang, Junzhou},
  journal={Medical Image Analysis},
  volume={65},
  pages={101789},
  year={2020},
  publisher={Elsevier}
}

@inproceedings{yao_deep_2019,
 author = {Yao, Jiawen and Zhu, Xinliang and Huang, Junzhou},
 booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
 copyright = {All rights reserved},
 pages = {496--504},
 publisher = {Springer},
 title = {Deep Multi-instance Learning for Survival Prediction from Whole Slide Images},
 year = {2019}
}

```
