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

device_dict = {}

try:
    import pynvml

    device_dict.update({"NvidiaGPU": {"device_id": 0}})
except:
    print("pynvml not found")

try:
    from jtop import jtop

    device_dict.update({"JetsonDev": {}})
except:
    print("jetson_stats not found")

calculator = Calculator(device_dict)


prune_list = [
    ["mv", "gdf", "S2", "uniform", "0.35"],
    ["ef", "gdf", "S2", "uniform", "0.36"],
    ["pf", "gdf", "S2", "uniform", "0.45"],
    ["mv", "gdf", "S2", "mmu", "0.35"],
    ["ef", "gdf", "S2", "mmu", "0.36"],
    ["pf", "gdf", "S2", "mmu", "0.55"],
]


@calculator.measure(times=2000, warmup=1000)
def inference(model, example_input):
    model(*example_input)


device = torch.device("cuda:0")


def Achelous_prune_energy(backbone, neck, phi, pm, pa):
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi=phi,
        backbone=backbone,
        neck=neck,
        spp=True,
        nano_head=False,
    )
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
        pm,
        pa,
        "cpu",
        imt_dict,
        bmt_dict,
        ["image_radar_encoder.radar_encoder.rc_blocks.0.weight_conv1"],
    )

    MACs, params = cal_flops(model, example_input, "cpu")
    example_input[0] = example_input[0].to(device)
    example_input[1] = example_input[1].to(device)
    model.to(device)
    model.eval()

    inference(model, example_input)
    res = calculator.summary(verbose=False)
    return (
        res[0],
        res[1],
        res[4],
        MACs,
        params,
    )  # power (W), energy (J), FPS, MACs, params


if __name__ == "__main__":
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
    results = []
    for backbone, neck, phi, pm, pa in prune_list:
        (
            power,
            energy,
            FPS,
            MACs,
            params,
        ) = Achelous_prune_energy(backbone, neck, phi, pm, pa)
        # print(
        #     "backbone: {}, neck: {}, phi: {}, pm: {}, pa: {}, power: {}, energy: {}, FPS: {}, MACs: {}, params: {}".format(
        #         backbone, neck, phi, pm, pa, power, energy, FPS, MACs, params
        #     )
        # )
        results.append(
            [
                backbone,
                neck,
                phi,
                pm,
                pa,
                power,
                energy,
                FPS,
                MACs,
                params,
            ]
        )
    os.mkdir("energy_output")
    np.savetxt(
        "energy_output/Achelous_prune_energy.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="backbone,neck,phi,pm,pa,power(W),energy(J),FPS,MACs,params",
    )
