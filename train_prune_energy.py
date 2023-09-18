# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.Achelous import *
from loss.detection_loss import (
    ModelEMA,
    YOLOLoss,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)
from utils.callbacks import LossHistory, EvalCallback
from utils_seg.callbacks import EvalCallback as EvalCallback_seg
from utils_seg_line.callbacks import EvalCallback as EvalCallback_seg_line
from utils.dataloader import YoloDataset, yolo_dataset_collate, yolo_dataset_collate_all
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils_seg.callbacks import LossHistory as LossHistory_seg
from utils_seg_line.callbacks import LossHistory as LossHistory_seg_line
from utils_seg_pc.callbacks import LossHistory as LossHistory_seg_pc
from utils_seg_pc.callbacks import EvalCallback as EvalCallback_seg_pc
import argparse
import torch_pruning as tp
import pruner.utils as pruner_utils

# pruning
from pruner.algorithm import prune_model, test_speed
import backbone.attention_modules.shuffle_attention as sa
import backbone.radar.RadarEncoder as re
from backbone.conv_utils.dcn import DeformableConv2d
from backbone.conv_utils.ghost_conv import GhostModule
from backbone.attention_modules.eca import eca_block
import backbone.vision.mobilevit_modules.mobilevit as mv
import backbone.vision.ImageEncoder as IE
import backbone.vision.poolformer_modules.poolformer as pf
from nets.Achelous import Achelous3T
from pruner.node import (
    MVitNode,
    DCNNode,
    ShuffleAttnNode,
    GhostModuleNode,
    ecaNode,
    Attn4DNode,
    Attn4DDownNode,
    # emLayerNormNode,
    # emPosEncNode,
)
from unip.utils.evaluation import cal_fps, cal_flops
from unip.utils.energy import Calculator

device_dict = {
    "NvidiaDev": {"device_id": 0},
}
calculator = Calculator(device_dict)

if __name__ == "__main__":
    # =========== 参数解析实例 =========== #
    parser = argparse.ArgumentParser()

    # 添加参数解析
    parser.add_argument("--fp16", type=str, default="True")
    parser.add_argument("--backbone", type=str, default="en")
    parser.add_argument("--neck", type=str, default="gdf")
    parser.add_argument("--nd", type=str, default="True")
    parser.add_argument("--phi", type=str, default="S0")
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--nw", type=int, default=4)
    parser.add_argument("--spp", type=str, default="True")
    parser.add_argument("--pm", default=0, type=float, help="pruning amount")
    parser.add_argument("--pa", default="uniform", type=str, help="pruning algroithm")

    args = parser.parse_args()

    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = True if args.fp16 == "True" else False

    # ------------------------------------------------------#
    #   backbone (4 options): ef (EfficientFormer), en (EdgeNeXt), ev (EdgeViT), mv (MobileViT)
    # ------------------------------------------------------#
    backbone = args.backbone

    # ------------------------------------------------------#
    #   neck (2 options): gdf (Ghost-Dual-FPN), cdf (CSP-Dual-FPN)
    # ------------------------------------------------------#
    neck = args.neck

    # ------------------------------------------------------#
    #   spp: True->SPP, False->SPPF
    # ------------------------------------------------------#
    spp = True if args.spp == "True" else False

    # ------------------------------------------------------#
    #   detection head (2 options): normal -> False, lightweight -> True
    # ------------------------------------------------------#
    lightweight = True if args.nd == "True" else False

    # ------------------------------------------------------#
    #   input_shape     all models support 320*320, all models except mobilevit support 416*416
    # ------------------------------------------------------#
    input_shape = [args.resolution, args.resolution]
    # ------------------------------------------------------#
    #   The size of model, three options: S0, S1, S2
    # ------------------------------------------------------#
    phi = args.phi
    # ------------------------------------------------------#

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi=phi,
        backbone=backbone,
        neck=neck,
        spp=spp,
        nano_head=False,
    ).to(device)

    model.cpu()
    example_input = [
        torch.randn(1, 3, 320, 320, requires_grad=True),
        torch.randn(1, 3, 320, 320, requires_grad=True),
    ]
    imt_dict = {
        GhostModule: GhostModuleNode,
        eca_block: ecaNode,
        DeformableConv2d: DCNNode,
        sa.ShuffleAttention: ShuffleAttnNode,
        # for mobilevit
        mv.Transformer: MVitNode,
        # for efficientformer
        IE.Attention4D: Attn4DNode,
        IE.Attention4DDownsample: Attn4DDownNode,
        # # for edgenext
        # em.layers.LayerNorm: emLayerNormNode,
        # em.layers.PositionalEncodingFourier: emPosEncNode,
    }
    bmt_dict = {
        IE.AttnFFN: [
            ["token_mixer", 0, "layer_scale_1", 0],
            ["mlp.norm2", 0, "layer_scale_2", 0],
        ],
        IE.FFN: [["mlp.norm2", 0, "layer_scale_2", 0]],
        pf.PoolFormerBlock: [
            ["norm1", 0, "layer_scale_1", 0],
            ["norm2", 0, "layer_scale_2", 0],
        ],
    }
    prune_model(
        model,
        example_input,
        args.pm,
        args.pa,
        "cpu",
        imt_dict,
        bmt_dict,
        ["image_radar_encoder.radar_encoder.rc_blocks.0.weight_conv1"],
    )

    example_input = [
        torch.randn(1, 3, 320, 320).to(device),
        torch.randn(1, 3, 320, 320).to(device),
    ]

    @calculator.measure(times=2000, warmup=1000)
    def inference(model, example_input):
        model(*example_input)

    model.to(device)
    model.eval()
    with torch.no_grad():
        inference(model, example_input)

    power, energy, _, _ = calculator.summary()
    # print(f"{backbone}-{neck}-{phi}-{args.pa}-{args.pm}, power(mW), energy(J)")
    import os

    os.system(
        f"echo '{backbone}-{neck}-{phi}-{args.pa}-{args.pm}, {power}, {energy}' >> energy.csv"
    )
    print(f"{backbone}-{neck}-{phi}-{args.pa}-{args.pm}, {power}, {energy}")
