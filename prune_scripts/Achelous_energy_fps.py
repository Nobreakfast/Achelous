import os
import platform

if platform.system() != "Linux":
    exit(0)
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import unip
from unip.utils.energy import Calculator
from unip.utils.evaluation import cal_flops
from nets.Achelous import *

device_dict = {}

try:
    import pynvml

    device_dict.update({"NvidiaDev": {"device_id": 0}})
except:
    print("pynvml not found")

try:
    from jtop import jtop

    device_dict.update({"JetsonDev": {}})
except:
    print("jetson_stats not found")

calculator = Calculator(device_dict)


phi_list = ["S0", "S1", "S2"]
backbone_list = [["mv", "ef", "en", "ev", "rv", "pf"], ["mo", "fv"]]
neck_list = [["gdf", "cdf"], ["rdf"]]


@calculator.measure(times=2000, warmup=1000)
def inference(model, example_input):
    model(*example_input)


device = torch.device("cuda:0")


def Achelous_energy(phi, backbone, neck):
    example_input = [torch.randn(1, 3, 320, 320), torch.randn(1, 3, 320, 320)]
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

    if neck == "rdf":
        for child in model.children():
            if isinstance(child, nn.Module):
                print("deploy:", type(child).__name__)
                child.deploy = True

        for module in model.modules():
            if hasattr(module, "reparameterize"):
                print("reparameterize:", type(module).__name__)
                module.reparameterize()

    MACs, params = cal_flops(model, example_input, "cpu")
    example_input[0] = example_input[0].to(device)
    example_input[1] = example_input[1].to(device)
    model.to(device)
    model.eval()

    inference(model, example_input)
    res = calculator.summary(verbose=False)
    return (
        res[0],
        res[1] / 2e3,
        res[4],
        MACs,
        params,
    )  # power (W), energy (J), FPS, MACs, params


if __name__ == "__main__":
    print("=" * 20, "test_BasePruner_with_Achelous", "=" * 20)
    results = []
    for phi in phi_list:
        for i in range(2):
            for backbone in backbone_list[i]:
                for neck in neck_list[i]:
                    (
                        power,
                        energy,
                        FPS,
                        MACs,
                        params,
                    ) = Achelous_energy(phi, backbone, neck)
                    results.append(
                        [
                            backbone,
                            neck,
                            phi,
                            power,
                            energy,
                            FPS,
                            MACs,
                            params,
                        ]
                    )
    # save results to csv
    results = np.array(results)
    # make dir energy_output
    os.system("mkdir -p energy_output")
    np.savetxt(
        "energy_output/Achelous_energy.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="backbone,neck,phi,power(W),energy(J),FPS,MACs,params",
    )
