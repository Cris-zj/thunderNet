import torch
import pdb
from cpp import libpsroi as psroi
from torch import nn
from torch.autograd import Function

class PSROI(Function):
    @staticmethod
    def forward(ctx,fsam,box,alpha,p):
        fsam_shape=fsam.shape
        ctx.save_for_backward(box)
        ctx.saved_shape=fsam_shape
        ctx.constant=(alpha,p)
        ret=psroi.PSROI_Forward(fsam,box,alpha,p)
        ret=ret.view(-1,alpha*p*p)
        return ret
    
    @staticmethod
    def backward(ctx, grad_outputs):
        box=ctx.saved_tensors[0]
        n,c,w,h=ctx.saved_shape
        alpha,p=ctx.constant
        grad=psroi.PSROI_Backward(grad_outputs,box,alpha,p,w,h)
        return grad,None,None,None

class PSROI_layer(nn.Module):
    def __init__(self,alpha,p):
        super(PSROI_layer,self).__init__()
        self.alpha=alpha
        self.p=p

    def forward(self,fsam,box):
        return PSROI.apply(fsam,box,self.alpha,self.p)

