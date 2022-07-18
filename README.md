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
For evaluating the performance of the proposed method, three challenging tasks in medical image segmentaion has been considered. In bellow, results of the proposed approach for synapse multi-organ segmentation is illustrated.
</br>

#### Performance Comparision on Synapse Multi-Organ Segmentation


|**Setting**| DSC   | HD | Aorta | Gallbladder | Kidney(L) | Kidney(R)| Liver | Pancreas| Spleen | Stomach |
| ------------------------------------------------------------------------------ |:----------------:|:---------------:|:-------:|:-----------:|:---------:|:----------------:|:-------:|:----------------:|:-------:|:-------:|
| **CNN as Encoder**                                                             | 75.89          | 28.87         | 85.03 | 65.17     | 80.18   | 76.38          | 90.49 | 57.29          | 85.68 | 69.93 |
| **Basic Scale Fusion**                                                         | 79.16          | 22.14         | 85.44 | 68.05     | 82.77   | 80.79          | 93.80 | 58.74          | 87.78 | 75.96 |
| **SSPP Level 1**                                                               | 79.01         | 26.63         | 85.61 | 68.47     | 82.43   | 78.02         | 94.19 | 58.52          | 88.34 | 76.46 |
| **SSPP Level 2**                                                               | 80.16          | 21.25         | 86.04 | 69.16     | 84.08   | 79.88          | 93.53 | 61.19          | 89.00 | 78.40 |
| **SSPP Level 3**                                                               | 79.87          | 18.93         | 86.34 | 66.41    | 84.13   | 82.40          | 93.73 | 59.28          | 89.66 | 76.99 |
| **SSPP Level 4**                                                               | 79.85          | 25.69         | 85.64 | 69.36     | 82.93   | 81.25          | 93.09 | 63.18          | 87.80 | 75.56 |


#### Perceptual visualization results on test data

![Synapse Multi-Organ Segmentation result](https://github.com/amirhossein-kz/HiFormer/blob/main/Figures/synapse.png)

### Model weights
You can download the learned weights for sypanse dataset in the following table. 

Dataset |Learned weights
------------ | -------------
[Synapse Multi-Organ Segmentation]() |[HiFormer-S](https://drive.google.com/file/d/1yyRyStyOkfQEKRiz64D6VaPiNPmzDkFJ/view?usp=sharing)
[Synapse Multi-Organ Segmentation]() |[HiFormer-B](https://drive.google.com/file/d/1-EV0szMsK4flOIu4BOc20mZEW7Nos4cU/view?usp=sharing)
[Synapse Multi-Organ Segmentation]() | [HiFormer-L](https://drive.google.com/file/d/12ADXxcy__9fB1nHo-6cSwLIWj8rJgN2o/view?usp=sharing)



### Query
All implementations are done by Amirhossein Kazerouni, Milad Soltany and Moein Heidari. For any query please contact us for more information.

```python
amirhossein477@gmail.com
soltany.m.99@gmail.com
moeinheidari7829@gmail.com

```

