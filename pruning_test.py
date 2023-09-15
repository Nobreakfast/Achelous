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
        torch.randn(1, 3, 320, 320, requires_grad=True).to(device),
        torch.randn(1, 3, 320, 320, requires_grad=True).to(device),
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
    sparsity = 0.98
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


def __test_onnx():
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

    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    )

    import deform_conv2d_onnx_exporter

    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
    example_input = [
        torch.randn(1, 3, 320, 320),
        torch.randn(1, 3, 320, 320),
    ]
    torch.onnx.export(
        model,
        (example_input[0], example_input[1]),
        "achelous.onnx",
        input_names=["image", "radar"],
        output_names=["det_head0", "det_head1", "det_head2", "se_seg", "lane_seg"],
        opset_version=12,
    )
    import onnx
    import onnxruntime

    onnx_model = onnxruntime.InferenceSession("achelous.onnx")
    onnx_x = [i.numpy() for i in example_input]
    onnx_result = onnx_model.run(
        ["det_head0", "det_head1", "det_head2", "se_seg", "lane_seg"],
        {"image": onnx_x[0], "radar": onnx_x[1]},
    )
    for i in range(5):
        print(onnx_result[i].shape, onnx_result[i].sum())
    model.eval()
    norm_result = model(*example_input)
    print(norm_result[0][0].shape, norm_result[0][0].sum())
    print(norm_result[0][1].shape, norm_result[0][1].sum())
    print(norm_result[0][2].shape, norm_result[0][2].sum())
    print(norm_result[1].shape, norm_result[1].sum())
    print(norm_result[2].shape, norm_result[2].sum())


def __test_onnx_radar():
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

    import deform_conv2d_onnx_exporter

    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S2",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    ).image_radar_encoder.radar_encoder
    # write custom onnx op for deformable conv
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args

    example_input = torch.rand(1, 3, 320, 320)

    torch.onnx.export(
        model,
        example_input,
        "radar.onnx",
        input_names=["radar"],
        output_names=["stage3", "stage4", "stage5"],
        opset_version=12,
    )

    # load onnx model
    import onnx
    import onnxruntime

    onnx_model = onnxruntime.InferenceSession("radar.onnx")
    onnx_input_name = onnx_model.get_inputs()[0].name
    onnx_output_name = onnx_model.get_outputs()[0].name
    onnx_x = example_input.numpy()
    onnx_result = onnx_model.run(["stage3", "stage4", "stage5"], {"radar": onnx_x})[0]
    print(onnx_result.shape, onnx_result.sum())
    model.eval()
    norm_result = model(example_input)[0].detach().numpy()
    print(norm_result.shape, norm_result.sum())


def __test_onnx_dcn():
    import torch
    import torchvision

    class dcn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 18, 3)
            self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

        def forward(self, x):
            return self.conv2(x, self.conv1(x))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = dcn()
            # self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.conv2(self.conv1(x))

    import deform_conv2d_onnx_exporter

    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

    model = Model()
    input = torch.rand(1, 3, 10, 10)
    torch.onnx.export(model, input, "dcn.onnx", opset_version=12)

    # load onnx model
    import onnx
    import onnxruntime

    onnx_model = onnxruntime.InferenceSession("dcn.onnx")
    onnx_input_name = onnx_model.get_inputs()[0].name
    onnx_output_name = onnx_model.get_outputs()[0].name
    onnx_x = input.numpy()
    onnx_result = onnx_model.run([onnx_output_name], {onnx_input_name: onnx_x})[0]
    print(onnx_result.shape, onnx_result.sum())
    norm_result = model(input).detach().numpy()
    print(norm_result.shape, norm_result.sum())


def __test_onnx_resnet50():
    import torch
    import torchvision

    model = torchvision.models.resnet50(pretrained=True)
    input = torch.rand(1, 3, 224, 224)
    torch.onnx.export(
        model,
        input,
        "resnet50.onnx",
        opset_version=12,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )

    # load onnx model
    import onnx
    import onnxruntime

    with torch.no_grad():
        onnx_model = onnxruntime.InferenceSession("resnet50.onnx")
        onnx_input_name = onnx_model.get_inputs()[0].name
        onnx_output_name = onnx_model.get_outputs()[0].name
        onnx_x = input.numpy()
        onnx_result = onnx_model.run([onnx_output_name], {onnx_input_name: onnx_x})[0]
        print(onnx_result.shape, onnx_result.sum())
        model.eval()
        norm_result = model(input).detach().numpy()
        print(norm_result.shape, norm_result.sum())


def __test_achelous_mmu():
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
        torch.randn(1, 3, 320, 320, requires_grad=True).to(device),
        torch.randn(1, 3, 320, 320, requires_grad=True).to(device),
    ]
    # test_speed(model, example_input, test_epoch, dev)
    model.cpu()
    example_input = [i.cpu() for i in example_input]
    prune_model(
        model, example_input, sparsity, "mmu", dev, imt_dict, bmt_dict, imk, skl
    )
    model.to(device)
    example_input = [i.to(device) for i in example_input]
    # test_speed(model, example_input, test_epoch, dev)


def main():
    # __test_example()
    # __test_resnet18()
    # __test_mvit()
    # __test_achelous()
    __test_achelous_mmu()
    # __test_achelous_radar()
    # __test_onnx()
    # __test_onnx_radar()
    # __test_onnx_dcn()
    # __test_onnx_resnet50()


if __name__ == "__main__":
    main()
