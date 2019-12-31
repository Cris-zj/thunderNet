import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from config import Config as cfg
from model import *
from common import *
from tools import PSROI_layer 
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils import data as Data
import os,pdb
from skimage import io,transform
import visdom
import cv2
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

def showImg(img,boxes,gt_box):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    if(boxes.ndim==1):boxes=boxes.reshape(-1,4)
    for i in range(len(boxes)):
        box=boxes[i]
        rect=plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False, edgecolor = 'red',linewidth=1)
        ax.add_patch(rect)
    for i in range(len(gt_box)):
        box=gt_box[i]
        rect=plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False, edgecolor = 'g',linewidth=1)
        ax.add_patch(rect)
    plt.show()

class ImageSet(Dataset):
    def __init__(self,root,path):
        self.root=root
        self.path=os.path.join(root,path)
        F=open(self.path,"r")
        self.lines=F.readlines()
        self.rows=cfg.input_hight
        self.cols=cfg.input_width
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):
        line=self.lines[index].split(" ")
        assert(len(line)>=7)
        imgPath=line[0]
        # print(imgPath)
        length=int(line[1])
        label=[]
        boxes=[]
        for i in range(length):
            label.append(int(line[2+i*5]))
            box=[]
            for j in range(4):
                box.append( int(line[3+i*5+j]) )
            boxes.append(box)
        boxes=np.array(boxes,dtype=np.int)
        label=np.array(label,dtype=np.int)
        img=io.imread(os.path.join(self.root,imgPath))
        # io.imsave(os.path.join("/home/huangeryu/Desktop/test",imgPath),img)
        H,W,C=img.shape
        scale1,scale2=self.rows/H,self.cols/W
        scale=min(scale1,scale2)
        destImage=np.zeros((self.rows,self.cols,C),dtype=np.float32)
        ystart=int((self.rows-scale*H)/2);yend=ystart+int(scale*H)
        xstart=int((self.cols-scale*W)/2);xend=xstart+int(scale*W)
        img=transform.resize(img,(int(H*scale),int(W*scale),C),mode="constant",anti_aliasing=True)
        boxes=boxes*scale+[xstart,ystart,xstart,ystart]
        destImage[ystart:yend,xstart:xend]=(img-np.array(cfg.pix_mean))/cfg.std
        #-------------------------------
        # anchorbox,validindex=gen_anchor(cfg.anchor,cfg.feature_size,cfg.feature_strid)
        # validanchor=anchorbox
        # boxes_tensor=torch.from_numpy(boxes)
        # iou=iou2d(boxes_tensor,validanchor)
        # indices=np.where(iou>0.45)[1]
        # print(len(indices))
        # print(boxes.shape)
        # print(iou.max(dim=1))
        # if(len(indices)==0):
        #     indices=iou.argmax(dim=1)
        # showImg(destImage,validanchor[indices],boxes)
        #-----------------------------------------
        destImage=destImage.transpose((2,0,1))
        image=torch.from_numpy(destImage)
        # pdb.set_trace()
        return image,boxes,label
        
class ThunderNet(nn.Module):
    def __init__(self,anchor_num,class_num,forwardOnly=False):
        super(ThunderNet,self).__init__()
        self.backbone=Shufflev2Net(132,stage5=False)
        self.cem=CEM((264,528,528),cfg.thin_channels)
        self.rpn=RPN(cfg.thin_channels,anchor_num)
        self.sam=SAM(256,cfg.thin_channels)
        self.subnet=SubNet(cfg.thin_channels,class_num)
        self.psroi_layer=PSROI_layer(cfg.alpha,cfg.feature_roi_size)
        self.baseAnchor,self.validAnchorIndex=gen_anchor(cfg.anchor,cfg.feature_size,cfg.feature_strid)
        self.forwardOnly=forwardOnly

    def forward(self,x,gt_box,gt_label):
        c4,c5,cglb=self.backbone(x)
        fcem=self.cem(c4,c5,cglb)
        frpn,rpn_cls,rpn_reg=self.rpn(fcem)
        fsam=self.sam(fcem,frpn)
        if self.forwardOnly:
            rois=choose_roi_for_subnet_only_forward(rpn_cls,rpn_reg,cfg.num_propose_test,cfg.num_subnet_test,self.baseAnchor,self.validAnchorIndex,
                cfg.propose_nms_threshold,cfg.input_hight,cfg.input_width)
        else:    
            rois,labels=choose_roi_for_subnet(gt_box,rpn_cls,rpn_reg,cfg.num_propose,cfg.num_propose_roi,self.baseAnchor,self.validAnchorIndex,
                cfg.propose_nms_threshold,cfg.pos_roi_threshold,cfg.pos_subnet_loss,cfg.neg_subnet_loss,cfg.input_hight,
                cfg.input_width,cfg.ignore_width)
        boxes=warp_rois_for_psroi(rois,cfg.feature_strid)
        froi=self.psroi_layer(fsam,boxes)
        sub_cls,sub_reg=self.subnet(froi)
        if(self.forwardOnly): return sub_cls,sub_reg,rois
        length=rpn_cls.numel()//2
        table=choose_box_for_rpn_loss(gt_box,self.baseAnchor,self.validAnchorIndex,length,
                cfg.num_rpn_loss,cfg.neg_rpn_threshold,cfg.pos_rpn_threshold)
        rpn_cls_loss,rpn_loc_loss=clc_rpn_loss(rpn_cls,rpn_reg,table,gt_box,self.baseAnchor)
        subnet_cls_loss,subnet_loc_loss=clc_subnet_loss(sub_cls,sub_reg,labels,rois,gt_box,gt_label,cfg.feature_strid)
        return rpn_cls_loss,rpn_loc_loss,subnet_cls_loss,subnet_loc_loss

class Solver:
    def __init__(self,model,trainLoader,testLoader,epoch,path="./"):
        self.model=model
        self.trainLoader=trainLoader
        self.testLoader=testLoader
        self.epoch=epoch
        self.optimizer=optim.SGD(self.init_lr(),momentum=cfg.momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_step, gamma=cfg.gamma)
        self.path=path
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_interval=cfg.test_interval
        self.saved_interval=cfg.saved_interval
        self.vis=visdom.Visdom(env="tundernet")
        self.start_index=1
        self.lamda1=cfg.rpn_cls_weight
        self.lamda2=cfg.rpn_loc_weight
        self.lamda3=cfg.subnet_cls_weight
        self.lamda4=cfg.subnet_loc_weight

    def init_lr(self):
        params=[]
        for key,value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if cfg.finetuning:
                    unfine=np.array([key.find(item)!=-1 for item in cfg.keywords]).any()
                    if(unfine):
                        if "bias" in key:
                            params.append({"params":[value],"lr":cfg.new_lr,"weight_decay":0})
                        else:
                            params.append({"params":[value],"lr":cfg.new_lr,"weight_decay":cfg.weight_decay})
                    else:
                        if "bias" in key:
                            params.append({"params":[value],"lr":cfg.pre_lr,"weight_decay":0})
                        else:
                            params.append({"params":[value],"lr":cfg.pre_lr,"weight_decay":cfg.weight_decay})
                else:
                    if "bias" in key:
                        params.append({"params":[value],"lr":cfg.lr*2,"weight_decay":0})
                        # nn.init.constant_(value,cfg.bias)
                    else:
                        params.append({"params":[value],"lr":cfg.lr,"weight_decay":cfg.weight_decay})
                        # nn.init.normal_(value,cfg.mean,cfg.std)
        return params

    def train(self,index)->float:
        self.model.train()
        sum_loss=np.array([0.0],dtype=float)
        for batch_idx,(data,gt_box,gt_label) in enumerate(self.trainLoader):
            data=data.to(self.device)
            self.optimizer.zero_grad()
            rpn_cls_loss,rpn_loc_loss,subnet_cls_loss,subnet_loc_loss=self.model(data,gt_box,gt_label)
            loss=self.lamda1*rpn_cls_loss+self.lamda2*rpn_loc_loss+self.lamda3*subnet_cls_loss+(self.lamda4*subnet_loc_loss if subnet_loc_loss is not None else 0.0)
            loss.backward()
            self.optimizer.step()
            print("train:epoch={} {}/{} loss={} ({:.6},{:.6}) ({:.6},{:.6})".format(index,batch_idx*len(data),len(self.trainLoader.dataset),loss,
                rpn_cls_loss,rpn_loc_loss,subnet_cls_loss,subnet_loc_loss if subnet_loc_loss is not None else 0.0))
            self.vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win="train_{}".format(index),name="train",update="append",opts=dict(showlegend=True,title="train_{}".format(index)))
            sum_loss+=loss.item()
        self.vis.line(X=np.array([index]),Y=np.array([sum_loss.mean()/len(self.trainLoader)]),win="loss",name="train",update="append",opts=dict(showlegend=True))
        return loss.item()
        
    def test(self,index):
        with torch.no_grad():
            self.model.eval()
            testLoss=0.
            for data,gt_box,gt_label in self.testLoader:
                data=data.to(self.device)
                rpn_cls_loss,rpn_loc_loss,subnet_cls_loss,subnet_loc_loss=self.model(data,gt_box,gt_label)
                output=self.lamda1*rpn_cls_loss+self.lamda2*rpn_loc_loss+self.lamda3*subnet_cls_loss+(self.lamda4*subnet_loc_loss if subnet_loc_loss is not None else 0.0)
                testLoss+=output.item()
            testLoss/=len(self.testLoader.dataset)
            self.vis.line(X=np.array([index]),Y=np.array([testLoss]),win="loss",name="test",update="append",opts=dict(showlegend=True))
            print("test:{} Average loss:{} ({:.6},{:.6}) ({:.6},{:.6})".format(len(self.testLoader.dataset),testLoss,
                        rpn_cls_loss,rpn_loc_loss,subnet_cls_loss,subnet_loc_loss if subnet_loc_loss is not None else 0.0 ))
            return testLoss

    def __call__(self):
        for index in range(self.start_index,self.epoch+1):
            self.vis.line(X=np.array([index]),Y=np.array([self.scheduler.get_lr()[0]]),win="lr",update="append")
            self.train(index)
            if(index%self.test_interval==0):self.test(index)
            if(index%self.saved_interval==0):self.save(index)
            self.scheduler.step()
        print("done")

    def save(self,index):
        if index is None:
            index=self.epoch
        torch.save({"epoch":index,"state_dict":self.model.state_dict(),\
            "optimizer":self.optimizer.state_dict()},"{}model{}.pth.tar".format(self.path,index))
        print("{}model{}.pth.tar saved".format(self.path,index))

    def load(self,modeDir):
        checkpoint=torch.load(modeDir)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_index=checkpoint["epoch"]

    def continue_train(self,modeDir):
        self.load(modeDir)
        self.__call__()
    
LABEL_NAMES = np.array([
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'])

LABEL_NAMES2=np.array(["plate"])

def main():
    root="/home/huangeryu/Desktop/image"
    train="trainval.txt"
    val="test.txt"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader=Data.DataLoader(ImageSet(root,train),batch_size=1,shuffle=True)
    test_loader=Data.DataLoader(ImageSet(root,val),batch_size=1)
    net=ThunderNet(len(cfg.anchor),cfg.database_class_num).to(device)
    solver=Solver(net,train_loader,test_loader,cfg.epoch)
    solver.continue_train("./model11.pth.tar")
    # solver()
    solver.save(None)
    #-------------------------------------------
    # root='/home/huangeryu/Desktop/test'
    # fileList=os.listdir(root)
    # imagelist=[item for item in fileList if item.endswith('.jpg')]
    # for item in imagelist:
    #     imgPath=os.path.join(root,item)
    #     forwardOnly(imgPath,"./model5.pth.tar",cfg.input_hight,cfg.input_width)
    

def forwardOnly(imgPath,modeDir,rows=320,cols=320):
    #加载图片
    image=io.imread(imgPath)
    H,W,C=image.shape
    scale1,scale2=rows/H,cols/W
    scale=min(scale1,scale2)
    destImage=np.zeros((rows,cols,C),dtype=np.float32)
    ystart=int((rows-scale*H)/2);yend=ystart+int(scale*H)
    xstart=int((cols-scale*W)/2);xend=xstart+int(scale*W)
    img=transform.resize(image,(int(H*scale),int(W*scale),C),mode="constant",anti_aliasing=True)
    destImage[ystart:yend,xstart:xend]=(img-np.array(cfg.pix_mean))/cfg.std
    destImage=np.array(destImage,dtype=np.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input=torch.from_numpy(destImage.transpose(2,0,1).reshape(1,C,rows,cols)).to(device)
    #加载模型
    net=ThunderNet(len(cfg.anchor),cfg.database_class_num,True).to(device)
    checkPoint=torch.load(modeDir)
    model=checkPoint["state_dict"]
    net.load_state_dict(model)
    #模型推断
    with torch.no_grad():
        net.eval()
        sub_cls,sub_reg,rois=net(input,None,None)
    #结果处理
    sub_cls=F.softmax(sub_cls,dim=1).cpu()
    cls_result=torch.argmax(sub_cls,dim=1)
    rois=rois.reshape(-1,4).cpu()
    sub_reg=sub_reg.cpu()
    pred_box=loc2box2d(sub_reg,rois)
    test=box2loc2d(pred_box,rois)
    # pdb.set_trace()
    pred_box[:,2:]=(np.minimum(pred_box[:,2:],[xend-1,yend-1])-torch.tensor([xstart,ystart]))/scale
    pred_box[:,:2]=(np.maximum(pred_box[:,:2],[xstart,ystart])-torch.tensor([xstart,ystart]))/scale
    rois[:,2:]=(np.minimum(rois[:,2:],[xend-1,yend-1])-torch.tensor([xstart,ystart]))/scale
    rois[:,:2]=(np.maximum(rois[:,:2],[xstart,ystart])-torch.tensor([xstart,ystart]))/scale
    pred_box=pred_box.to(dtype=torch.float32)
    img=cv2.imread(imgPath)
    name=imgPath[imgPath.rfind("/")+1:]
    pos_index=torch.where(cls_result>0)[0]
    indices=np.arange(len(pos_index)).reshape(-1,1)
    scores=sub_cls[pos_index,cls_result[pos_index]]
    cls_result=cls_result[pos_index]
    pos_box=pred_box[pos_index]
    selectedIndex=np.where(scores>0.5)[0]
    scores=scores[selectedIndex]
    pos_box=pos_box[selectedIndex]
    cls_result=cls_result[selectedIndex]
    nms_index=ops.nms(pos_box,scores,0.3)
    for i in nms_index:
        box=pos_box[i]
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,255,0))
        cv2.putText(img,"{}".format(LABEL_NAMES[cls_result[i]-1]),(box[0]+20,box[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),1)
    cv2.imwrite("./.test/{}".format(name),img)
    showImg(img,rois,[])
    pdb.set_trace()

if __name__=="__main__":
    main()