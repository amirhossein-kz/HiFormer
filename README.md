# [HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation](https://arxiv.org/pdf/2207.08518.pdf), [WACV 2023](https://wacv2023.thecvf.com/home)

Hierarchical Multi-scale Representations for medical image segmentation with state of the art results on synapse multi-organ segmentation, skin lesion segmentation, multiple myeloma cell segmentation. This method designs two multi-scale feature representations using the seminal Swin Transformer module and a CNN-based encoder, then using a Double-Level Fusion (DLF) module, it allows a fine fusion of global and local features.
If this code helps with your research please consider citing the following paper:
</br>
> [Moein Heidari](https://scholar.google.com/citations?user=mir8D5UAAAAJ&hl=en&oi=sra)\*,
[Amirhossein Kazerouni](https://scholar.google.com/citations?user=aKDCc3MAAAAJ&hl=en)\*, [Milad Soltany](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=Gm23tVgAAAAJ)\*, [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Ehsan Khodapanah Aghdam](https://scholar.google.com/citations?user=a4DcyOYAAAAJ&hl=en), [Julien Cohen-Adad](https://scholar.google.ca/citations?user=6cAZ028AAAAJ&hl=en) and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation", download [link](https://arxiv.org/pdf/2207.08518).


#### Please consider starring us, if you found it useful. Thanks!

## Updates
- October 10, 2022: Accepted in WACV 2023!
- July 14, 2022: First release (Complete implemenation for [Synapse Multi-Organ Segmentation](https://www.synapse.org/#!Synapse:syn3193805/wiki/) dataset.)

This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch

## Prepare data, Train/Test
Please go to ["Instructions.ipynb"](https://github.com/amirhossein-kz/HiFormer/blob/main/Instructions.ipynb) for complete detail on dataset preparation and Train/Test procedure or follow the instructions below. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amirhossein-kz//HiFormer/blob/main/Instructions.ipynb)


1) Download the Synapse Dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

2) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

3) Run the below code to train HiFormer on the synapse dataset.

    ```bash
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5  --model_name hiformer-b --batch_size 10 --eval_interval 20 --max_epochs 400 
   ```
    **--root_path**     [Train data path]

    **--test_path**     [Test data path]

    **--eval_interval** [Evaluation epoch]

    **--model_name**    [Choose from [hiformer-s, hiformer-b, hiformer-l]]
    
4) Run the below code to test HiFormer on the synapse dataset.
    ```bash
    python test.py --test_path ./data/Synapse/test_vol_h5 --model_name hiformer-b --is_savenii --model_weight './hiformer-b_best.pth'
    ```
    **--test_path**     [Test data path]
    
    **--model_name**    [choose from [hiformer-s, hiformer-b, hiformer-l]]
    
    **--is_savenii**    [Whether to save results during inference]
    
    **--model_weight**  [HiFormer trained model path]


## Quick Overview
![Diagram of the proposed method](https://github.com/amirhossein-kz/HiFormer/blob/main/Figures/Model%20Overview.png)

## Results
For evaluating the performance of the proposed method, three challenging tasks in medical image segmentaion has been considered. In bellow, results of the proposed approach for synapse multi-organ segmentation is illustrated.
</br>

#### Performance Comparision on Synapse Multi-Organ Segmentation

| <h3 align="left">**Methods** </h3> | <p>DSC&#8593;</p> | <p>HD&#8595;</p>  |  <p>Aorta</p> | <p>Gallbladder</p> | <p>Kidney(L)</p> | <p>Kidney(R)</p> | <p>Liver</p> | <p>Pancreas</p> | <p>Spleen</p> | <p>Stomach</p> |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DARR |  69.77 |  -  |  74.74 |  53.77 |  72.31 | 73.24  |  94.08  |  54.18 | 89.90 |  45.96 |
| R50 U-Net |  74.68  |  36.87  |  87.74 |  63.66 |  80.60 |  78.19 |  93.74 | 56.90 |  85.87 | 74.16 |
| U-Net |  76.85 |  39.70 |  89.07 |  69.72 |  77.77 |  68.60 |  93.43 |  53.98 |  86.67 | 75.58 |
| R50 Att-UNet |  75.57 |  36.97 |  55.92 | 63.91 | 79.20 | 72.71 | 93.56 | 49.37 | 87.19 | 74.95 |
| Att-UNet |  77.77 |  36.02 | **89.55**  | 68.88 | 77.98 | 71.11 | 93.57 | 58.04 | 87.30 | 75.75 |
| R50 ViT |  71.29 |  32.87 |  73.73 |  55.13 |  75.80 |  72.20 |  91.51 |  45.99 |  81.99 | 73.95 |
| TransUnet |  77.48 |  31.69 |  87.23 |  63.13 |  81.87 |  77.02 |  94.08 |  55.86 |  85.08 |  75.62 |
| SwinUnet |  79.13 |  21.55 |  85.47 |  66.53 |  83.28 |  79.61 | 94.29 | 56.58 | 90.66| 76.60 |
| LeVit-Unet |  78.53 |  16.84 |  78.53 |  62.23 |  84.61 |  **80.25** | 93.11 | 59.07 | 88.86 | 72.76 |
| DeepLabv3+ (CNN) | 77.63 | 39.95 | 88.04 | 66.51 | 82.76 | 74.21 | 91.23 | 58.32 | 87.43 | 73.53 |
| **HiFormer-S** | 80.29 | 18.85 | 85.63 | **73.29** | 82.39 | 64.84 | 94.22  |**60.84** | **91.03** |  78.07|
| **HiFormer-B** | 80.39 | **14.70** | 86.21 | 65.69 | **85.23** | 79.77 | **94.61**  | 59.52 | 90.99 |  81.08|
| **HiFormer-L** | **80.69** | 19.14 | 87.03 | 68.61 | 84.23 | 78.37 | 94.07  | 60.77 | 90.44 |  **82.03**|


#### Perceptual visualization results on test data

![Synapse Multi-Organ Segmentation result](https://github.com/amirhossein-kz/HiFormer/blob/main/Figures/synapse.png)

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
* [CrossViT](https://github.com/IBM/CrossViT)

## Query
All implementations are done by Amirhossein Kazerouni, Milad Soltany and Moein Heidari. For any query please contact us for more information.

[*amirhossein477@gmail.com*](mailto:amirhossein477@gmail.com)

[*soltany.m.99@gmail.com*](mailto:soltany.m.99@gmail.com)

[*moeinheidari7829@gmail.com*](mailto:moeinheidari7829@gmail.com)


## Citation
```
@article{heidari2022hiformer,
  title={HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation},
  author={Heidari, Moein and Kazerouni, Amirhossein and Soltany, Milad and Azad, Reza and Aghdam, Ehsan Khodapanah and Cohen-Adad, Julien and Merhof, Dorit},
  journal={arXiv preprint arXiv:2207.08518},
  year={2022}
}
```
