from genericpath import exists
import ml_collections
import os
import wget

def get_swin_res34_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256, 512]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
    os.makedirs('./weights', exist_ok=True)

    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # Cross Attention Config
    cfg.depth = ([1, 3, 1], [1, 3, 1])
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 4.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = False
    cfg.qk_scale = None

    return cfg

def get_swin_res50_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024,2048]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
    os.makedirs('./weights', exist_ok=True)
    
    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'
    
    # Cross Attention Config
    cfg.depth = ([1, 3, 1], [1, 3, 1])
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 4.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = False
    cfg.qk_scale = None

    return cfg

def get_swin_res18_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet18"
    cfg.cnn_pyramid_fm  = [64, 128, 256, 512]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
    os.makedirs('./weights', exist_ok=True)
    
    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')    
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # Cross Attention Config
    cfg.depth = ([1, 3, 1], [1, 3, 1])
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 4.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = False
    cfg.qk_scale = None
    
    return cfg