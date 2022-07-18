# [HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation]()

Hierarchical Multi-scale Representations for medical image segmentation with state of the art results on synapse multi-organ segmentation, skin lesion segmentation, multiple myeloma cell segmentation. This method designs two multi-scale feature representations using the seminal Swin Transformer module and a CNN-based encoder, then using a Double-Level Fusion (DLF) module, it allows a fine fusion of global and local features.
If this code helps with your research please consider citing the following paper:
</br>
> [Moein Heidari](https://scholar.google.com/citations?user=mir8D5UAAAAJ&hl=en&oi=sra), [Amirhossein Kazerouni](https://scholar.google.com/citations?user=aKDCc3MAAAAJ&hl=en), [Milad Soltany](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=Gm23tVgAAAAJ), [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Ehsan Khodapanah Aghdam](https://scholar.google.com/citations?user=a4DcyOYAAAAJ&hl=en), [Julien Cohen-Adad](https://scholar.google.ca/citations?user=6cAZ028AAAAJ&hl=en) and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation", download [link]().


#### Please consider starring us, if you found it useful. Thanks

## Updates
- July 14, 2022: First release (Complete implemenation for [Synapse Multi-Organ Segmentation](https://www.synapse.org/#!Synapse:syn3193805/wiki/) ,[SKin Lesion Segmentation on ISIC 2017](https://challenge.isic-archive.com/landing/2017/), [SKin Lesion Segmentation on ISIC 2018](https://challenge2018.isic-archive.com/), [SKin Lesion Segmentation on PH2](https://www.fc.up.pt/addi/ph2%20database.html) and [Multiple Myeloma Cell Segmentation (SegPC 2021)](https://www.kaggle.com/sbilab/segpc2021dataset) dataset added.)

This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch

## Prepare data, Train/Test
Please go to ["Instructions.ipynb"](https://github.com/amirhossein-kz/HiFormer/blob/main/Instructions.ipynb) for complete detail on dataset preparation and Train/Test procedure. 

## Quick Overview
![Diagram of the proposed method](https://github.com/amirhossein-kz/HiFormer/blob/main/Figures/Model%20Overview.png)

## Results
For evaluating the performance of the proposed method, Three challenging tasks in medical image segmentaion has been considered. In bellow, results of the proposed approach for synapse multi-organ segmentation is illustrated.
</br>
#### Synapse Multi-Organ Segmentation


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
All implementations are done by Amirhossein Kazerouni, Milad Soltany and Moein Heidari. For any query please contact us for more information.

```python
moeinheidari7829@gmail.com

```

