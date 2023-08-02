import torch
import torch.nn as nn
import torch.nn.functional as F
from pruner.algorithm import prune_model, test_speed
from pruner import mvit
from pruner.node import MVitNode


def __test_example():
    from pruner.test_model import ExampleModel

    model = ExampleModel().eval()
    prune_model(model, torch.randn(1, 3, 4, 4), 0.7, "random", "cpu")


def __test_resnet18():
    from torchvision.models import resnet18

    model = resnet18()
    prune_model(model, torch.randn(1, 3, 32, 32), 0.7, "cpu")


def __test_mvit():
    from pruner.mvit import mobilevit_xxs

    model = mobilevit_xxs()
    imt_dict = {mvit.Transformer: MVitNode}
    prune_model(model, torch.randn(1, 3, 320, 320), 0.7, "cpu", imt_dict)


def __test_achelous():
    import backbone.attention_modules.shuffle_attention as sa
    import backbone.radar.RadarEncoder as re
    from backbone.conv_utils.dcn import DeformableConv2d
    from backbone.conv_utils.ghost_conv import GhostModule
    from backbone.attention_modules.eca import eca_block
    import backbone.vision.mobilevit_modules.mobilevit as mv
    import backbone.vision.ImageEncoder as IE
    import backbone.vision.edgenext_modules as em
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
        emLayerNormNode,
        emPosEncNode,
    )

    test_epoch = 1
    sparsity = 0.5
    # backbone = ["mv", "ef", "pf"]
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",  # FIXME: en, ev, rv
        neck="gdf",
        spp=True,
        nano_head=False,
    )
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
    # DONE: pruning bundling
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
    imk = [
        "image_radar_encoder.radar_encoder.rc_blocks.0.weight_conv1",
    ]
    skl = [
        # "lane_seg",
        # "se_seg",
        # "det_head",
    ]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    model.to(device).eval()
    example_input = [
        torch.randn(1, 3, 320, 320).to(device),
        torch.randn(1, 3, 320, 320).to(device),
    ]
    # test_speed(model, example_input, test_epoch, dev)
    model.cpu()
    example_input = [i.cpu() for i in example_input]
    prune_model(
        model, example_input, sparsity, "erk", dev, imt_dict, bmt_dict, imk, skl
    )
    model.to(device)
    example_input = [i.to(device) for i in example_input]
    # test_speed(model, example_input, test_epoch, dev)


def __test_achelous_radar():
    import backbone.attention_modules.shuffle_attention as sa
    import backbone.radar.RadarEncoder as re
    from backbone.conv_utils.dcn import DeformableConv2d
    from backbone.conv_utils.ghost_conv import GhostModule
    from backbone.attention_modules.eca import eca_block
    import backbone.vision.mobilevit_modules.mobilevit as mv
    import backbone.vision.ImageEncoder as IE
    import backbone.vision.edgenext_modules as em
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
        emLayerNormNode,
        emPosEncNode,
    )

    test_epoch = 1
    sparsity = 0.5
    # backbone = ["mv", "ef", "pf"]
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",  # FIXME: en, ev, rv
        neck="gdf",
        spp=True,
        nano_head=False,
    )
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
    # DONE: pruning bundling
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
    imk = [
        "rc_blocks.0.weight_conv1",
    ]
    skl = [
        # "lane_seg",
        # "se_seg",
        # "det_head",
    ]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    model.to(device).eval()
    example_input = [
        torch.randn(1, 3, 320, 320).to(device),
    ]
    # test_speed(model, example_input, test_epoch, dev)
    model.cpu()
    example_input = [i.cpu() for i in example_input]
    prune_model(
        model.image_radar_encoder.radar_encoder,
        example_input,
        sparsity,
        "erk",
        dev,
        imt_dict,
        bmt_dict,
        imk,
        skl,
    )
    model.to(device)
    example_input = [i.to(device) for i in example_input]
    # test_speed(model, example_input, test_epoch, dev)


def main():
    # __test_example()
    # __test_resnet18()
    # __test_mvit()
    # __test_achelous()
    __test_achelous_radar()


if __name__ == "__main__":
    main()
