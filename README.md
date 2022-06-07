# SwinRetina paper

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
model_name = ['Swin_Res34', 'Swin_Res50', 'Swin_Res18']
