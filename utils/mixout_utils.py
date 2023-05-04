##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional
from collections import OrderedDict
from torch.autograd.function import InplaceFunction


class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns 
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p) 
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None. 
    # I modified the code of dropout in PyTorch. 
    @staticmethod
    def _make_noise(input:torch.Tensor) -> torch.Tensor:
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, 
                ctx, 
                input:torch.Tensor, 
                target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
                p:float=0.0, 
                training:bool=False, 
                inplace:bool=False) -> torch.Tensor:

        if p < 0 or p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {p}")

        if target is not None and input.size() != target.size():
            raise ValueError(f"A target tensor size must match with a input tensor size {input.size()}, but got {target.size()}")
        
        ctx.p = p    
        ctx.training = training
        
        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        
        if ctx.p == 0 or not ctx.training:
            return output
        
        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], *([1] * (len(input.size())-1)))
        ctx.noise.expand_as(input)
        
        if ctx.p == 1:
            output = target.clone()
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        
        return output


    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Optional[torch.Tensor]:
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input:torch.Tensor, 
           target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
           p:float=0.0, 
           training:bool=False, 
           inplace:bool=False) -> torch.Tensor:

    return Mixout.apply(input, target, p, training, inplace)

class MixLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p)) 
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer, 
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p) 
    def __init__(self, 
                in_features:int, 
                out_features:int, 
                bias:bool=True, 
                target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
                p:float=0.0) -> None:

        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.target = target
        self.p = p

        if self.p < 0 or self.p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {self.p}")
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return F.linear(input, mixout(self.weight, self.target, 
                                      self.p, self.training), self.bias)

    def extra_repr(self):
        type_ = 'drop' if self.target is None else 'mix'
        type_ += "out" 
        return f'{type_}={self.p}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This example script created by Michael Wilson
## Department of Linguistics, Yale University
## Email: michael.a.wilson@yale.edu
## GitHub: mawilson1234
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from torch import nn
from transformers import AutoModelForMaskedLM
from copy import deepcopy

def replace_layer_for_mixout(module: nn.Module, mixout_prob: float) -> nn.Module:
    '''
    Replaces a single layer with the correct layer for use with Mixout.
    If module is nn.Dropout, replaces it with a Dropout where p = 0.
    If module is nn.Linear, replaces it with a MixLinear where p(mixout) = mixout_prob.
    In all other cases, returns the module unchanged.
    
        params:
            module (nn.Module)    : a module to replace for Mixout
            mixout_prob (float)   : the desired Mixout probability
        
        returns:
            module (nn.Module)    : the module set up for use with Mixout
    '''
    # if isinstance(module, nn.Dropout):
    #     return nn.Dropout(0)
    if isinstance(module, nn.Linear):
        target_state_dict   = deepcopy(module.state_dict())
        bias                = True if module.bias is not None else False
        new_module          = MixLinear(
                                module.in_features,
                                module.out_features,
                                bias,
                                target_state_dict['weight'],
                                mixout_prob
                            )
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module

def recursive_setattr(obj: 'any', attr: str, value: 'any') -> None:
    '''
    Recursively sets attributes for objects with children.
    
        params:
            obj (any)   : the object with children whose attribute is to be set
            attr (str)  : the (nested) attribute of the object, with levels indicated by '.'
                            for instance attr='attr1.attr2' sets the attr2 of obj.attr1 to
                            the passed value
            value (any) : what to set the attribute to
    '''
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)