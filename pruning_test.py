import torch
import torch.nn as nn
import torch.nn.functional as F
from pruner.algorithm import prune_model, test_speed
from pruner import mvit
from pruner.node import MVitNode


def __test_example():
    from pruner.test_model import ExampleModel

    model = ExampleModel().eval()
    prune_model(model, torch.randn(1, 3, 4, 4), 0.7, "cpu")


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
    from nets.Achelous import Achelous3T
    from pruner.node import MVitNode, DCNNode, ShuffleAttnNode, GhostModuleNode, ecaNode

    test_epoch = 1
    sparsity = 0.76
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
    imt_dict = {
        mv.Transformer: MVitNode,
        DeformableConv2d: DCNNode,
        sa.ShuffleAttention: ShuffleAttnNode,
        GhostModule: GhostModuleNode,
        eca_block: ecaNode,
    }
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
        model,
        example_input,
        sparsity,
        "featio50",
        dev,
        imt_dict,
        ["image_radar_encoder.radar_encoder.rc_blocks.0.weight_conv1"],
    )
    model.to(device)
    example_input = [i.to(device) for i in example_input]
    # test_speed(model, example_input, test_epoch, dev)


def main():
    # __test_example()
    # __test_resnet18()
    # __test_mvit()
    __test_achelous()


if __name__ == "__main__":
    main()
