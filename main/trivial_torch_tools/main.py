import numpy
import torch
import torch.nn as nn
from .generics import product, bundle, large_pickle_save, large_pickle_load
from .core import device, to_tensor
from .misc import layer_output_shapes
from .model import init, convert_args
from .one_hots import OneHotifier

import functools
class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__(*args)
        self.input_shape = kwargs.get("input_shape", None)
    
    def forward(self, neuron_activations):
        return functools.reduce((lambda x, each_layer: each_layer.forward(x)), self.children(), neuron_activations)
    
    @property
    def layer_shapes(self):
        return layer_output_shapes(self, self.input_shape)
    
    @property
    def output_shape(self):
        return self.input_shape if len(self) == 0 else self.layer_shapes[-1]
    
    @property
    def output_size(self):
        total = 1
        for each in self.output_shape:
            total *= each
        return total