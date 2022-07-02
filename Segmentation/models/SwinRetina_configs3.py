import ml_collections
import os


def get_swin_dense121_cv_120_221_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet121"
    cfg.cnn_pyramid_fm  = [256, 512, 1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
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


def get_swin_dense169_cv_120_221_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet169"
    cfg.cnn_pyramid_fm  = [256, 512, 1280]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
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

def get_swin_dense201_cv_120_221_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet201"
    cfg.cnn_pyramid_fm  = [256, 512, 1792]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
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

# ----------------------------------------------------------------------------------#

def get_swin_dense121_cv_130_331_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet121"
    cfg.cnn_pyramid_fm  = [256, 512, 1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
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


def get_swin_dense169_cv_130_331_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet169"
    cfg.cnn_pyramid_fm  = [256, 512, 1280]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
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

def get_swin_dense201_cv_130_331_66_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet201"
    cfg.cnn_pyramid_fm  = [256, 512, 1792]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
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

# ----------------------------------------------------------------------------------#

def get_swin_dense121_cv_110_111_33_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet121"
    cfg.cnn_pyramid_fm  = [256, 512, 1024]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # Cross Attention Config
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


def get_swin_dense169_cv_110_111_33_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet169"
    cfg.cnn_pyramid_fm  = [256, 512, 1280]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # Cross Attention Config
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg

def get_swin_dense201_cv_110_111_33_true_cfg():
    cfg = ml_collections.ConfigDict()
    cfg.cnn_backbone = "densenet201"
    cfg.cnn_pyramid_fm  = [256, 512, 1792]
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9

    # custom
    cfg.resnet_pretrained = True
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # Cross Attention Config
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg