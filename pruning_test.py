import torch
import torch.nn as nn
import torch.nn.functional as F
from pruner.backward2node import prune_model
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
    prune_model(
        model,
        [torch.randn(1, 3, 320, 320), torch.randn(1, 3, 320, 320)],
        0.7,
        "cpu",
        imt_dict,
    )


def main():
    # __test_example()
    # __test_resnet18()
    # __test_mvit()
    __test_achelous()


if __name__ == "__main__":
    main()
