# [Contextual Attention Network: Transformer Meets U-Net](https://arxiv.org/abs/2203.01932)

Contexual attention network for medical image segmentation with state of the art results on skin lesion segmentation, multiple myeloma cell segmentation. This method incorpotrates the transformer module into a U-Net structure so as to concomitantly capture long-range dependency along with resplendent local informations.
If this code helps with your research please consider citing the following paper:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Moein Heidari](https://scholar.google.com/citations?user=mir8D5UAAAAJ&hl=en&oi=sra), [Yuli Wu](https://scholar.google.com/citations?user=qlun0AgAAAAJ) and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "Contextual Attention Network: Transformer Meets U-Net", download [link](https://arxiv.org/abs/2203.01932).

```python
@article{reza2022contextual,
  title={Contextual Attention Network: Transformer Meets U-Net},
  author={Reza, Azad and Moein, Heidari and Yuli, Wu and Dorit, Merhof},
  journal={arXiv preprint arXiv:2203.01932},
  year={2022}
}

```

#### Please consider starring us, if you found it useful. Thanks

## Updates
- February 27, 2022: First release (Complete implemenation for [SKin Lesion Segmentation on ISIC 2017](https://challenge.isic-archive.com/landing/2017/), [SKin Lesion Segmentation on ISIC 2018](https://challenge2018.isic-archive.com/), [SKin Lesion Segmentation on PH2](https://www.fc.up.pt/addi/ph2%20database.html) and [Multiple Myeloma Cell Segmentation (SegPC 2021)](https://www.kaggle.com/sbilab/segpc2021dataset) dataset added.)

This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch


## Run Demo
For training deep model and evaluating on each data set follow the bellow steps:</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `train_skin.py` for training the model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing segmentation result, run `evaluate_skin.py`. It will represent performance measures and will saves related results in `results` folder.</br>

**Notice:**
For training and evaluating on ISIC 2017 and ph2 follow the bellow steps :

**ISIC 2017**- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18\7`. </br> then Run ` 	Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>
**ph2**- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` 	Prepare_ph2.py` for data preperation and dividing data to train,validation and test sets. </br>
Follow step 3 and 4 for model traing and performance estimation. For ph2 dataset you need to first train the model with ISIC 2017 data set and then fine-tune the trained model using ph2 dataset.



## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/TMUnet/blob/main/Figures/Method%20(Main).png)

### Perceptual visualization of the proposed Contextual Attention module.
![Diagram of the proposed method](https://github.com/rezazad68/TMUnet/blob/main/Figures/Method%20(Submodule).png)


## Results
For evaluating the performance of the proposed method, Two challenging task in medical image segmentaion has been considered. In bellow, results of the proposed approach illustrated.
</br>
#### Task 1: SKin Lesion Segmentation


#### Performance Comparision on SKin Lesion Segmentation
In order to compare the proposed method with state of the art appraoches on SKin Lesion Segmentation, we considered Drive dataset.  

Methods (On ISIC 2017) |Dice-Score | Sensivity| Specificaty| Accuracy
------------ | -------------|----|-----------------|---- 
Ronneberger and et. all [U-net](https://arxiv.org/abs/1505.04597)       |0.8159	  |0.8172  |0.9680  |0.9164	  
Oktay et. all [Attention U-net](https://arxiv.org/abs/1804.03999)   |0.8082  |0.7998      |0.9776	  |0.9145
Lei et. all [DAGAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300803)   |0.8425	  |0.8363       |0.9716	 |0.9304
Chen et. all [TransU-net](https://arxiv.org/abs/2102.04306)   |0.8123  |0.8263     |0.9577	  |0.9207
Asadi et. all [MCGU-Net](https://arxiv.org/abs/2003.05056)   |0.8927	  |	0.8502      |**0.9855**	  |0.9570	
Valanarasu et. all [MedT](https://arxiv.org/abs/2102.10662)   |0.8037	  |0.8064       |0.9546	  |0.9090
Wu et. all [FAT-Net](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003728)   |0.8500	  |0.8392  |0.9725	  |0.9326
Azad et. all [Proposed TMUnet](https://arxiv.org/abs/2203.01932)	  |**0.9164** 	| **0.9128**	|0.9789	  |**0.9660**
### For more results on ISIC 2018 and PH2 dataset, please refer to [the paper](https://arxiv.org/abs/2203.01932)


#### SKin Lesion Segmentation segmentation result on test data

![SKin Lesion Segmentation  result](https://github.com/rezazad68/TMUnet/blob/main/Figures/Skin%20lesion_segmentation.png)
(a) Input images. (b) Ground truth. (c) [U-net](https://arxiv.org/abs/2102.10662). (d) [Gated Axial-Attention](https://arxiv.org/abs/2102.10662). (e) Proposed method without a contextual attention module and (f) Proposed method.


## Multiple Myeloma Cell Segmentation

#### Performance Evalution on the Multiple Myeloma Cell Segmentation task

Methods | mIOU
------------ | -------------
[Frequency recalibration U-Net](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Azad_Deep_Frequency_Re-Calibration_U-Net_for_Medical_Image_Segmentation_ICCVW_2021_paper.pdf)	 |0.9392 
[XLAB Insights](https://arxiv.org/abs/2105.06238)	|0.9360
[DSC-IITISM](https://arxiv.org/abs/2105.06238)	|0.9356	  
[Multi-scale attention deeplabv3+](https://arxiv.org/abs/2105.06238)	 |0.9065	  
[U-Net](https://arxiv.org/abs/1505.04597)	  |0.7665
[Baseline](https://arxiv.org/abs/2203.01932)	  |0.9172
[Proposed](https://arxiv.org/abs/2203.01932)	  |**0.9395**



#### Multiple Myeloma Cell Segmentation results

![Multiple Myeloma Cell Segmentation result](https://github.com/rezazad68/TMUnet/blob/main/Figures/Cell_segmentation.png)

### Model weights
You can download the learned weights for each dataset in the following table. 

Dataset |Learned weights
------------ | -------------
[ISIC 2018]() |[TMUnet](https://drive.google.com/file/d/1EU4stQUtUt6bcSoWswBYpfTZd53sVAJy/view?usp=sharing)
[ISIC 2017]() |[TMUnet](https://drive.google.com/file/d/1gEb8juWB2JjxAws91D3S0wxnrVwuMRZo/view?usp=sharing)
[Ph2]() | [TMUnet](https://drive.google.com/file/d/1soZ6UYhZk7r5-klflJHZxtbdH6pKi7t6/view?usp=sharing)



### Query
All implementations are done by Reza Azad and Moein Heidari. For any query please contact us for more information.

```python
rezazad68@gmail.com
moeinheidari7829@gmail.com

```

