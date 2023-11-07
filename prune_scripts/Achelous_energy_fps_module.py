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
    model(example_input)


device = torch.device("cuda:0")


def forward_hook_record_input(module, input, output):
    setattr(module, "input", input[0])


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

    hooks = []
    for m in model.modules():
        # if module is single module like conv, bn, relu, etc. ignore
        if not m._modules:
            continue
        hooks.append(m.register_forward_hook(forward_hook_record_input))
    model(*example_input)
    for h in hooks:
        h.remove()
    results = []
    for k, m in model.named_modules():
        if not m._modules:
            continue
        if not hasattr(m, "input"):
            continue
        try:
            example_input = torch.randn_like(m.input)
            m(example_input)
        except:
            continue
        MACs, params = cal_flops(m, example_input, "cpu")
        example_input = example_input.to(device)
        m.to(device)
        m.eval()
        inference(m, example_input)
        res = calculator.summary(verbose=False)
        results.append([res[0], res[1] / 2e3, res[4], MACs, params])
    results = np.array(results)
    np.savetxt(
        f"energy_output/modules/{backbone}-{neck}-{phi}.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="module,power(W),energy(J),MACs,flops,params",
    )


if __name__ == "__main__":
    # make dir energy_output
    os.system("mkdir -p energy_output/modules")
    for phi in phi_list:
        for i in range(2):
            for backbone in backbone_list[i]:
                for neck in neck_list[i]:
                    Achelous_energy(phi, backbone, neck)
