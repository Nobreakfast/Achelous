import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence
from structure import *

def get_sparsity(model, amount):
    """
    Get the sparsity dict for a model.
    """
    sparsity_dict = {}
    sum = 0.0
    for k, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("Conv2d layer", k)
            in_ch = m.in_channels
            out_ch = m.out_channels
            strides = m.stride
            featio = out_ch/strides[0]/strides[1]/in_ch/(torch.sum(m.weight.data).item()+1e-8)
        elif isinstance(m, nn.Linear):
            # print("Linear layer", k)
            in_ch = m.in_features
            out_ch = m.out_features
            featio = out_ch/in_ch/(torch.sum(m.weight.data).item()+1e-8)
        else: 
            continue
        sparsity_dict[k] = 1/featio
        sum += sparsity_dict[k]
    for k in sparsity_dict.keys():
        sparsity_dict[k] = amount*(1-sparsity_dict[k]/sum)
    return sparsity_dict

#def prune_model(model, sparsity_dict):
#    """
#    Prune the model according to the sparsity dict.
#    """
#    pass

def __list_to_dict(list: Sequence[str]) -> dict:
    """
    Convert a list to a dict.
    """
    dict = {}
    for k1 in list:
        dict[k1] = {}
        for k2 in list:
            dict[k1][k2] = 0
    return dict

    

def get_dep_dcit(model, example_input=torch.randn(2,3,32,32), type_list=[nn.Conv2d, nn.Linear, nn.BatchNorm2d], ignore_list=[]):
    """
    Get the dependency dict for a model.
    """
    # get the white list and module dict
    white_list = ['output']
    module_list = {}
    for k, m in model.named_modules():
        if not isinstance(m, tuple(type_list)):
            continue
        if k in ignore_list:
            continue
        white_list.append(k)
        module_list[k] = m
    
    # get the dependency dict
    dep_dict = __list_to_dict(white_list)
    
    # get the name dict for id address to key
    name_dict = {}
    for k, m in model.named_modules():
        name_dict[id(m)] = k

    # get module forward and backward dict
    forward_dict, backward_dict = {}, {}
    def __forward_hook_callback(module, input, output):
        forward_dict[name_dict[id(module)]] = {}
        forward_dict[name_dict[id(module)]]['output'] = output
        forward_dict[name_dict[id(module)]]['input'] = input[0]
    def __backward_hook_callback(module, grad_input, grad_output):
        backward_dict[name_dict[id(module)]] = {}
        backward_dict[name_dict[id(module)]]['output'] = grad_output[0]
        backward_dict[name_dict[id(module)]]['input'] = grad_input
    hooks = []
    for k, m in module_list.items():
        hooks.append(m.register_forward_hook(__forward_hook_callback))
        hooks.append(m.register_backward_hook(__backward_hook_callback))
    out = model(example_input)
    out.sum().backward()

    # fill the dependency dict
    def __check_id(g):
        for k, m in forward_dict.items():
            if m['output'].grad_fn == g:
                return k
        return 'none'
    def __fill_dep(parent, g, level=0): # TODO level: useless params
        if g == None: return
        # check current node name
        # if g.__class__.__name__ == 'AddBackward0':
        #     if parent[-4:] == '.add':
        #         id = parent[:-4]
        #     id = parent + '.add'
        # else:
        #     id = __check_id(g)
        id = __check_id(g)
        if id != 'none':
            dep_dict[parent][id] = 1
            dep_dict[id][parent] = -1
        else:
            # print('Warning: {} not in white list'.format(g.__class__.__name__))
            id = parent
        for subg in g.next_functions:
            __fill_dep(id, subg[0], level+1)
    __fill_dep('output', out.grad_fn, 0)
    for hook in hooks:
        hook.remove()
    return dep_dict, module_list, name_dict

def get_group_list(dep_dict, module_list, name_dict):
    related_dict = {}
    for k1, v1 in dep_dict.items():
        related_dict[k1] = {'in':[], 'out':[]}
        for k2, v2 in v1.items():
            if v2 == 1:
                related_dict[k1]['in'].append(k2)
            elif v2 == -1:
                related_dict[k1]['out'].append(k2)
            else:
                continue

    # get the group dict
    group_list = []
    layers = []
    # layers_next = []

    def __add_layer(m_list, layer_type):
        tmp_list = []
        for i in m_list:
            if i == 'output':
                continue
            if isinstance(module_list[i], nn.BatchNorm2d):
                tmp_list += __add_layer(related_dict[i][layer_type], layer_type)
            else:
                tmp_list.append(i)
        return tmp_list


    for k, v in related_dict.items():
        layer = __add_layer(v['in'], 'in')
        layers.append(layer)
        # layer_next = __add_layer(v['out'], 'out')
        # layers_next.append(layer_next)

    def __get_merged_data(layers):
        merged_data = []
        # iterate over each sublist in the layers
        for sublist in layers:
            sublist_set = set(sublist)
            merged = False

            temp = []
            for merged_sublist in merged_data:
                if sublist_set.intersection(set(merged_sublist)):
                    merged_sublist.extend(temp)
                    merged_sublist.extend(sublist)
                    merged_sublist[:] = list(set(merged_sublist))
                    merged = True
                    if temp is not []:
                        try:
                            merged_data.remove(temp)
                        except:
                            pass
                    temp = merged_sublist

            if not merged:
                merged_data.append(sublist)
        return list(filter(None, merged_data))
    
    layers = __get_merged_data(layers)
    # layers_next = __get_merged_data(layers_next)

    for i in layers:
        group = {'modules': [], 'next': []}
        for j in i:
            group['modules'].append(j)
            group['next'].extend(related_dict[j]['out'])
            for k in related_dict[j]['out']:
                if k == 'output':
                    continue
                if isinstance(module_list[k], nn.BatchNorm2d):
                    group['next'].extend(related_dict[k]['out'])
        group['next'] = list(set(group['next']))
        group_list.append(group)

    return group_list


def prune_model(model, ratio=0.5, prune_type='random'):
    dep_dict, module_list, name_dict = get_dep_dcit(model)
    group_list = get_group_list(dep_dict, module_list, name_dict)
    for group in group_list:
       print("modules:", group['modules'])
       print("next_modules:", group['next'])

    def __set_module(name, new_module):
        name_list = name.split('.')
        module = model
        for i in name_list[:-1]:
            if i.isdigit():
                module = module[int(i)]
            else:
                module = getattr(module, i)
        setattr(module, name_list[-1], new_module)
        

    for i in group_list:
        #if i['next'] == ['output']: # TODO: delete this and add in the get_group_list
        #    continue
        if isinstance(module_list[i['modules'][0]], nn.Conv2d):
            out_ch = module_list[i['modules'][0]].out_channels
        else:
            out_ch = module_list[i['modules'][0]].out_features

        if prune_type == 'random':
            prune_num = int(out_ch * ratio)
            prune_idx = torch.randperm(out_ch)[:prune_num]
        
        for j in i['modules']:
            if isinstance(module_list[j], nn.Conv2d):
                pruned_module = structure_conv(module_list[j], prune_idx, 0)
            elif isinstance(module_list[j], nn.Linear):
                pruned_module = structure_fc(module_list[j], prune_idx, 0)
            module_list[j] = pruned_module
            __set_module(j, pruned_module)

        for j in i['next']:
            if j == 'output':
                continue
            if isinstance(module_list[j], nn.Conv2d):
                pruned_module = structure_conv(module_list[j], prune_idx, 1)
            elif isinstance(module_list[j], nn.Linear):
                pruned_module = structure_fc(module_list[j], prune_idx, 1)
            elif isinstance(module_list[j], nn.BatchNorm2d):
                pruned_module = structure_bn(module_list[j], prune_idx)
            module_list[j] = pruned_module
            __set_module(j, pruned_module) #structure_conv(module_list[j], prune_idx, 1))
        
    for k, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            print(k, m)

    print(model(torch.randn(2,3,32,32)).shape)

    return model
        

def __test_get_dep_dict():
    from torchvision.models import resnet18
    # model = conv5()
    model = resnet18()
    dep_dict, module_list, name_dict = get_dep_dcit(model)
    # convert dict to pd
    import pandas as pd
    dep_df = pd.DataFrame(dep_dict)
    dep_df.to_csv('~/Downloads/dep.csv')

def __test_get_group_list():
    from torchvision.models import resnet18
    # model = conv5()
    model = resnet18()
    dep_dict, module_list, name_dict = get_dep_dcit(model)
    group_list = get_group_list(dep_dict, module_list, name_dict)
    for group in group_list:
       print("modules:", group['modules'])
       print("next_modules:", group['next'])

def __test_pruning_model():
    from torchvision.models import resnet18
    # model = conv5()
    model = resnet18()
    prune_model(model)

if __name__ == '__main__':
    # __test_get_dep_dict()
    # __test_get_group_list()
    __test_pruning_model()
