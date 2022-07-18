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

| <h3 align="left">**Methods** </h3> | DSC <p>&#8593;</p> | HD <p> &#8595;</p>  | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V-Net** |  68.81 |  -  |  75.34 |  51.87 |  77.10 | **80.75**  |  87.84  |  40.05 | 80.56 |  56.98 |
| **R50 U-Net** |  74.68  |  36.87  |  87.74 |  63.66 |  80.60 |  78.19 |  93.74 | 56.90 |  85.87 | 74.16 |
| **U-Net** |  76.85 |  39.70 |  89.07 |  **69.72** |  77.77 |  68.60 |  93.43 |  53.98 |  86.67 | 75.58 |
| **R50 Att-UNet** |  75.57 |  36.97 |  55.92 | 63.91 | 79.20 | 72.71 | 93.56 | 49.37 | 87.19 | 74.95 |
| **Att-UNet** |  77.77 |  36.02 | **89.55**  | 68.88 | 77.98 | 71.11 | 93.57 | 58.04 | 87.30 | 75.75 |
| **R50 ViT** |  71.29 |  32.87 |  73.73 |  55.13 |  75.80 |  72.20 |  91.51 |  45.99 |  81.99 | 73.95 |
| **TransUnet** |  77.48 |  31.69 |  87.23 |  63.13 |  81.87 |  77.02 |  94.08 |  55.86 |  85.08 |  75.62 |
| **SwinUnet** |  79.13 |  21.55 |  85.47 |  66.53 |  83.28 |  79.61 | **94.29** | 56.58 | **90.66** | 76.60 |
| **DeepLabv3+ (CNN)** | 77.63 | 39.95 | 88.04 | 66.51 | 82.76 | 74.21 | 91.23 | 58.32 | 87.43 | 73.53 |
| **TransDeepLab** | **80.16** | **21.25** | 86.04 | 69.16 | **84.08** | 79.88 | 93.53  |**61.19** | 89.00 |  **78.40**|

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
## Citation
```

```

