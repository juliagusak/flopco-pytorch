import torch
import torch.nn as nn

import numpy as np
from collections import defaultdict
from functools import partial
import copy

from flopco.compute_layer_flops import *


class FlopCo():
    
    def __init__(self, model, img_size = (1, 3, 224, 224), custom_tensor = None, device = 'cpu', instances = None):
        '''
        instances: list of layer types,
            supported types are [nn.Conv2d, nn.Linear,
            nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Softmax]
        '''
        self.device = device
        self.model = model

        self.img_size = img_size
        self.custom_tensor = custom_tensor

        self.input_shapes = None
        self.output_shapes = None
        
        self.flops = None
        self.macs = None
        
        self.params = None


        if instances is not None:
            self.instances = instances
        else:
            self.instances  = [nn.Conv2d,
                               nn.Linear
                              ]
#             self.instances  = [nn.Conv2d,
#                                nn.Linear,
#                                nn.BatchNorm2d,
#                                nn.ReLU,
#                                nn.MaxPool2d,
#                                nn.AvgPool2d,
#                                nn.Softmax
#                               ]
    
        self.ltypes = None
        self.get_ltypes()
        
        self.get_stats(shapes = True, flops = True, macs = True, params = True)
        
        self.total_flops = sum([sum(v) for v in self.flops.values()])
        self.total_macs = sum([sum(v) for v in self.macs.values()])
        
        self.total_params = sum(self.params.values())
        
        
        self.relative_flops = defaultdict(None,\
                                          {k: sum(v)/self.total_flops\
                                           for k,v in self.flops.items()})
        
        self.relative_macs = defaultdict(None,\
                                  {k: sum(v)/self.total_macs\
                                   for k,v in self.macs.items()})
        
        self.relative_params = defaultdict(None,\
                          {k: v/self.total_params\
                           for k,v in self.params.items()})

        del self.model
        torch.cuda.empty_cache()
        
        
    def __str__(self):
        print_info = "\n".join([str({k:v}) for k,v in self.__dict__.items()])
        
        return str(self.__class__) + ": \n" + print_info
        

    def get_ltypes(self):
        self.ltypes = defaultdict(defaultdict)
        
        for mname, m in self.model.named_modules():
            if m.__class__ in self.instances:
                
                self.ltypes[mname]['type'] = type(m)
                
                if isinstance(m, nn.Conv2d):
                    self.ltypes[mname]['kernel_size'] = m.kernel_size
                    self.ltypes[mname]['groups'] = m.groups                
    
    def count_params(self):
        self.params = defaultdict(int)
        
        for mname, m in self.model.named_modules():
            if m.__class__ in self.instances:
                
                self.params[mname] = 0
                
                for p in m.parameters():
                    self.params[mname] += p.numel()
              
    
    def _save_shapes(self, name, mod, inp, out):
        self.input_shapes[name].append(inp[0].shape)
        self.output_shapes[name].append(out.shape)
    
    def _save_flops(self, name, mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            flops = compute_conv2d_flops(mod, inp[0].shape, out.shape)
            
        elif isinstance(mod, nn.Linear):
            flops = compute_fc_flops(mod, inp[0].shape, out.shape)
            
        elif isinstance(mod, nn.BatchNorm2d):
            flops = compute_bn2d_flops(mod, inp[0].shape, out.shape)
            
        elif isinstance(mod, nn.ReLU):
            flops = compute_relu_flops(mod, inp[0].shape, out.shape)
        
        elif isinstance(mod, nn.MaxPool2d):
            flops = compute_maxpool2d_flops(mod, inp[0].shape, out.shape)
            
        elif isinstance(mod, nn.AvgPool2d):
            flops = compute_avgpool2d_flops(mod, inp[0].shape, out.shape)
            
        elif isinstance(mod, nn.Softmax):
            flops = compute_softmax_flops(mod, inp[0].shape, out.shape)

        else:
            flops = -1
        
        self.flops[name].append(flops)
        
        
    def _save_macs(self, name, mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            flops = compute_conv2d_flops(mod, inp[0].shape, out.shape, macs = True)
            
        elif isinstance(mod, nn.Linear):
            flops = compute_fc_flops(mod, inp[0].shape, out.shape, macs = True)
            
        elif isinstance(mod, nn.BatchNorm2d):
            flops = compute_bn2d_flops(mod, inp[0].shape, out.shape, macs = True)
            
        elif isinstance(mod, nn.ReLU):
            flops = compute_relu_flops(mod, inp[0].shape, out.shape, macs = True)
        
        elif isinstance(mod, nn.MaxPool2d):
            flops = compute_maxpool2d_flops(mod, inp[0].shape, out.shape, macs = True)
            
        elif isinstance(mod, nn.AvgPool2d):
            flops = compute_avgpool2d_flops(mod, inp[0].shape, out.shape, macs = True)
            
        elif isinstance(mod, nn.Softmax):
            flops = compute_softmax_flops(mod, inp[0].shape, out.shape, macs = True)

        else:
            flops = -1

        
        self.macs[name].append(flops)


    def get_stats(self, shapes = True, flops = False, macs = False, params = False):
        
        if params:
            self.count_params()
            
        if shapes:
            self.input_shapes = defaultdict(list)
            self.output_shapes = defaultdict(list)
       
        if flops:
            self.flops = defaultdict(list)
        
        if macs:
            self.macs = defaultdict(list)

        with torch.no_grad():
            for name, m in self.model.named_modules():
                to_compute = sum(map(lambda inst : isinstance(m, inst),
                                            self.instances))
                if to_compute:

                    if shapes:
                        m.register_forward_hook(partial(self._save_shapes, name))
                    
                    if flops:
                        m.register_forward_hook(partial(self._save_flops, name))
                    
                    if macs:
                        m.register_forward_hook(partial(self._save_macs, name))

            if self.custom_tensor is None:
                batch = torch.rand(*self.img_size).to(self.device)
                self.model(batch)
            else:
                batch = self.custom_tensor
                if isinstance(self.custom_tensor, list):
                    self.model(*batch)
                elif isinstance(self.custom_tensor, dict):
                    self.model(**batch)
                else:
                    raise TypeError(f'Input tensor should be of type list or dict, got {type(batch)}')

            batch = None

            for name, m in self.model.named_modules():
                m._forward_pre_hooks.clear()
                m._forward_hooks.clear()

        torch.cuda.empty_cache()