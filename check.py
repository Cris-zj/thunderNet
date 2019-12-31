import torch
from torch.autograd import gradcheck
import numpy as np 
from tools import PSROI
import pdb
import torchvision.ops as ops
batchsize=1
bndPerbath=64
p=7
k=5
fsam=torch.randn(batchsize,k*p*p,20,20,dtype=torch.float,requires_grad=True)
box=[]
for batch in range(batchsize):
    for boxth in range(bndPerbath):
        while True:
            x=np.random.randint(0,20)
            y=np.random.randint(0,20)
            w=np.random.randint(1,20)
            h=np.random.randint(1,20)
            if(x+w-1<20 and y+h-1<20):
                box.append([x,y,w,h])
                break
box1=torch.tensor(box,dtype=torch.int)
box_tensor=box1.view(batchsize,bndPerbath,4)
print(box_tensor)
input = (fsam,box_tensor,k,p)
ret=PSROI.apply(fsam,box_tensor,k,p)
ret.backward(torch.ones(batchsize*bndPerbath,k*p*p,dtype=torch.float))
# scores=torch.rand(batchsize*bndPerbath,dtype=torch.float).view(batchsize,bndPerbath,1)
# box1[:,2:]=box1[:,:2]+box1[:,2:]
# box1=box1.to(dtype=torch.float32).view(batchsize,bndPerbath,4)
# nmsret=ops.nms(box1[0],scores[0],0.5)
# iret=gradcheck(PSROI.apply,input,eps=1e-6, atol=1e-4)
# print(iret)
pdb.set_trace()
print(ret)
print(fsam.grad.sum(dim=(2,3)))