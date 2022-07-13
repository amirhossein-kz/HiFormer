from genericpath import exists
import ml_collections
import os
import wget

##### Swin_Res_34 #####
# Total Params: 23251475 
def get_swin_res34_cv_11_11_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 24752051
def get_swin_res34_cv_120_221_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg

# Total Params: 24142931
def get_swin_res34_cv_120_111_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 24752051
def get_swin_res34_cv_120_221_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 23251475 
def get_swin_res34_cv_11_11_44_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1]]
    cfg.num_heads = (4, 4)
    cfg.mlp_ratio=(1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 23251475 
def get_swin_res34_cv_11_11_33_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 23251475  
def get_swin_res34_cv_110_111_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 26835155
def get_swin_res34_cv_130_331_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 3, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(3., 3., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 23568179
def get_swin_res34_cv_110_221_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 29519219
def get_swin_res34_cv_140_441_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 4, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(4., 4., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


##### Swin_Res_50 #####
# Total Params: 24011539
def get_swin_res50_cv_110_111_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 25508083
def get_swin_res50_cv_120_221_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


def get_swin_res50_cv_120_221_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 27595219
def get_swin_res50_cv_130_331_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 3, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(3., 3., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 24898963
def get_swin_res50_cv_120_111_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


def get_swin_res50_cv_120_111_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# Total Params: 24011539
def get_swin_res50_cv_11_11_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


##### Swin_Res_18 #####
def get_swin_res18_cv_11_11_612_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "resnet18"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.swin_pyramid_fm = [96, 192, 384]
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
    cfg.depth = [[1, 1]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True
        
    return cfg