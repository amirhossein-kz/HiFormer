# SwinRetina paper

## Approach 2
Config 1:
```
$ cd Segmentation
$ pip install -r requirements.txt
$ python train_v2.py --root_path [train dataset path]/.../train_npz \
                     --test_path [test dataset path]/.../test_vol_h5 \
                     --batch_size 24 \
                     --eval_interval 20 \
                     --base_lr 0.001 \
                     --max_epochs 301 \
                     --model_name 'Swin_Res34' \

```

Config 2:
```
$ cd Segmentation
$ pip install -r requirements.txt
$ python train_v2.py --root_path [train dataset path]/.../train_npz \
                     --test_path [test dataset path]/.../test_vol_h5 \
                     --batch_size 24 \
                     --eval_interval 20 \
                     --base_lr 0.001 \
                     --max_epochs 301 \
                     --model_name 'Swin_Res50' \

```

Config 3:
```
$ cd Segmentation
$ pip install -r requirements.txt
$ python train_v2.py --root_path [train dataset path]/.../train_npz \
                     --test_path [test dataset path]/.../test_vol_h5 \
                     --batch_size 24 \
                     --eval_interval 20 \
                     --base_lr 0.001 \
                     --max_epochs 301 \
                     --model_name 'Swin_Res18' \

```

## Approach 1
Config 4:
```
$ cd Segmentation
$ pip install -r requirements.txt
$ python train.py --root_path [train dataset path]/.../train_npz \
                     --test_path [test dataset path]/.../test_vol_h5 \
                     --batch_size 24 \
                     --eval_interval 20 \
                     --base_lr 0.001 \
                     --max_epochs 301 \
                     --model_name 'Swin_Res34' \

```

Config 5:
```
$ cd Segmentation
$ pip install -r requirements.txt
$ python train_v2.py --root_path [train dataset path]/.../train_npz \
                     --test_path [test dataset path]/.../test_vol_h5 \
                     --batch_size 24 \
                     --eval_interval 20 \
                     --base_lr 0.001 \
                     --max_epochs 301 \
                     --model_name 'Swin_Res50' \

```

Config 6:
```
$ cd Segmentation
$ pip install -r requirements.txt
$ python train_v2.py --root_path [train dataset path]/.../train_npz \
                     --test_path [test dataset path]/.../test_vol_h5 \
                     --batch_size 24 \
                     --eval_interval 20 \
                     --base_lr 0.001 \
                     --max_epochs 301 \
                     --model_name 'Swin_Res18' \

```
