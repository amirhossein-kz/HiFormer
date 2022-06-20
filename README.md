# SwinRetina paper
All configurations are available in the Run.txt file.

Batch Size 10 works in Kaggle and Colab.

Config 1:
```
$ cd Segmentation
$ pip install -r requirements.txt
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_11_11_612_true_cfg' \
                    --num_workers 2 \
```

Config 2:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_120_221_612_true_cfg' \
                    --num_workers 2 \
```

Config 3:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_120_111_612_true_cfg' \
                    --num_workers 2 \
```

Config 4:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_120_221_66_true_cfg' \
                    --num_workers 2 \
```

Config 5:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_11_11_44_true_cfg' \
                    --num_workers 2 \
```

Config 6:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_11_11_33_true_cfg' \
                    --num_workers 2 \
```

Config 7:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_110_111_612_true_cfg' \
                    --num_workers 2 \
```

Config 8:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_130_331_66_true_cfg' \
                    --num_workers 2 \
```

Config 9:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_110_221_612_true_cfg' \
                    --num_workers 2 \
```

Config 10:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res34_cv_140_441_66_true_cfg' \
                    --num_workers 2 \
```

Config 11:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res50_cv_110_111_612_true_cfg' \
                    --num_workers 2 \
```

Config 12:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res50_cv_120_221_612_true_cfg' \
                    --num_workers 2 \
```

Config 13:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res50_cv_130_331_66_true_cfg' \
                    --num_workers 2 \
```

Config 14:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res50_cv_120_111_612_true_cfg' \
                    --num_workers 2 \
```

Config 15:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res50_cv_11_11_612_true_cfg' \
                    --num_workers 2 \
```

Config 16:
```
!python train_v3.py --root_path ./data/Synapse/train_npz \   # [train dataset path]
                    --test_path ./data/Synapse/test_vol_h5 \ # [test dataset path]
                    --batch_size 10 \
                    --eval_interval 20 \
                    --base_lr 0.01 \
                    --max_epochs 361 \
                    --model_name 'swin_res18_cv_11_11_612_true_cfg' \
                    --num_workers 2 \
```
