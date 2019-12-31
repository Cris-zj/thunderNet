import torch
import torch.nn as nn
import torchvision
import functools
import torch.nn.functional as F
import pdb
def channel_shuffle(x,g):
    n,c,w,h=x.shape
    x=x.view(n,g,c//2,w,h).permute(0,2,1,3,4).contiguous().view(n,c,h,w)
    return x

class ShuffleV2BasicUnit(nn.Module):
    def __init__(self,stride,in_channels,out_channels):
        super(ShuffleV2BasicUnit,self).__init__()
        self.stride=stride
        if(stride==2):
            branch_channels=out_channels//2
        else:
            in_channels=in_channels//2
            branch_channels=in_channels
        conv11=functools.partial(nn.Conv2d,kernel_size=1,stride=1,padding=0,bias=False)
        dw_conv55=functools.partial(nn.Conv2d,kernel_size=5,stride=stride,padding=2,bias=False)
        if(stride>1):
            self.branch1=nn.Sequential(dw_conv55(in_channels,in_channels,groups=in_channels),
                nn.BatchNorm2d(in_channels),
                conv11(in_channels,branch_channels),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        self.branch2=nn.Sequential(conv11(in_channels,branch_channels),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            dw_conv55(branch_channels,branch_channels,groups=branch_channels),
            nn.BatchNorm2d(branch_channels),
            conv11(branch_channels,branch_channels),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        if self.stride==1:
            x1,x2=x.chunk(2,dim=1)
            out=torch.cat((x1,self.branch2(x2)),dim=1)
        else:
            out=torch.cat((self.branch1(x),self.branch2(x)),dim=1)
        out=channel_shuffle(out,2)
        return out

class Shufflev2Net(nn.Module):
    def __init__(self,stage2_channels=132,stage5=True,stage5_channels=512):
        super(Shufflev2Net,self).__init__()
        self.exist_stage5=stage5
        stage_names=["stage{}".format(i) for i in range(2,5)]
        repeat_nums=[3,7,3]
        channel_nums=[stage2_channels*pow(2,i-2) for i in range(2,5)]
        in_channel=24
        for name,repeat_num,channel_num in zip(stage_names,repeat_nums,channel_nums):
            seq=[ShuffleV2BasicUnit(2,in_channel,channel_num)]
            for i in range(repeat_num):
                seq.append(ShuffleV2BasicUnit(1,channel_num,channel_num))
            setattr(self,name,nn.Sequential(*seq))
            in_channel=channel_num
        if(stage5):
            self.conv5=nn.Sequential(nn.Conv2d(in_channel,stage5_channels,1,1,0,bias=False),
                nn.BatchNorm2d(stage5_channels),
                nn.ReLU(inplace=True),
            )
            setattr(self,"stage5",self.conv5)
            in_channel=stage5_channels
        self.conv1=nn.Sequential(nn.Conv2d(3,24,3,2,1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool=nn.MaxPool2d(3,stride=2)
        self.fc=nn.Linear(in_channel,1000,bias=True)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.stage2(x)
        c3=self.stage3(x)
        c4=self.stage4(c3)
        if(self.exist_stage5):
            c5=self.stage5(c4)
            cglb=c5.mean([2,3]).view(c5.size(0),c5.size(1),1,1)
            return c4,c5,cglb
        else:
            cglb=c4.mean([2,3]).view(c4.size(0),c4.size(1),1,1)
            return c3,c4,cglb 

class CEM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CEM,self).__init__()
        assert(len(in_channels)==3)
        self.out_channels=out_channels
        self.C4_branch=nn.Conv2d(in_channels[0],out_channels,1,1,0,bias=True)
        
        self.C5_branch=nn.Sequential(nn.Conv2d(in_channels[1],out_channels,1,1,0,bias=True),
            nn.Upsample(scale_factor=2,mode="nearest"),
        )
        self.Cglb_branch=nn.Conv2d(in_channels[2],out_channels,1,1,0,bias=True)
    
    def forward(self,x_4,x_5,x_glb):
        x4_lat=self.C4_branch(x_4)
        x5_lat=self.C5_branch(x_5)
        xglb_lat=self.Cglb_branch(x_glb).view(-1,self.out_channels,1,1)
        return x4_lat+x5_lat+xglb_lat

class SAM(nn.Module):
    def __init__(self,fRPN_channels,out_channels):
        super(SAM,self).__init__()
        self.sam_branch=nn.Sequential(nn.Conv2d(fRPN_channels,out_channels,1,1,0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    def forward(self,fCEM,fRPN):
        return fCEM.mul(self.sam_branch(fRPN))

class RPN(nn.Module):
    def __init__(self,in_channels,anchor_num):
        super(RPN,self).__init__()
        self.dw_conv55=nn.Sequential(nn.Conv2d(in_channels,in_channels,5,1,2,groups=in_channels,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,256,1,1,0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.rpn_cls=nn.Conv2d(256,2*anchor_num,1,1,0,bias=True)
        self.rpn_reg=nn.Conv2d(256,4*anchor_num,1,1,0,bias=True)
    
    def forward(self,x):
        x=self.dw_conv55(x)
        rpn_cls=self.rpn_cls(x)
        rpn_reg=self.rpn_reg(x)
        return x,rpn_cls,rpn_reg

class SubNet(nn.Module):
    def __init__(self,in_channels,class_num):
        super(SubNet,self).__init__()
        self.fc=nn.Linear(in_channels,1024)
        self.cls=nn.Linear(1024,class_num)
        self.reg=nn.Linear(1024,4)
    
    def forward(self,x):
        x=self.fc(x)
        sub_cls=self.cls(x)
        sub_reg=self.reg(x)
        return sub_cls,sub_reg


