from genericpath import exists
import ml_collections
import os
import wget

def get_swin_res34_cfg():
    """Returns the ViT-L/16 cfguration."""
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256, 512]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained_path = True
    os.makedirs('./weights', exist_ok=True)
    if os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        cfg.pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'
    else:
        print('Swin-transformer model is downloading ...')    
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    return cfg

def get_swin_res50_cfg():
    """Returns the ViT-L/16 cfguration."""
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024,2048]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained_path = True
    os.makedirs('./weights', exist_ok=True)
    if os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        cfg.pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'
    else:
        print('Swin-transformer model is downloading ...')    
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")
    return cfg

def get_swin_res18_cfg():
    """Returns the ViT-L/16 cfguration."""
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet18"
    cfg.cnn_pyramid_fm  = [64, 128, 256, 512]
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained_path = True
    os.makedirs('./weights', exist_ok=True)
    if os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        cfg.pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'
    else:
        print('Swin-transformer model is downloading ...')    
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    return cfg