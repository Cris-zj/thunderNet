import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from eval import confusionMatrix as cfsm
import visdom
from early_stopping import EarlyStopping
import torch.utils.data as Data
import os
from config import Config as cfg
from skimage import io,transform,color
from model import SNet146
from model import Shufflev2Net
import pdb
import warnings

class dataset(Data.Dataset):
    def __init__(self,root,filename):
        self.root=root
        self.path=os.path.join(self.root,filename)
        with open(self.path,"r") as F:
            self.lines=F.readlines()
        self.rows=cfg.input_hight
        self.cols=cfg.input_width
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):
        line=self.lines[index].split(" ")
        imgPath=line[0]
        gt_label=int(line[1])
        img=io.imread(os.path.join(self.root,imgPath))
        if(img.ndim==2):
            H,W=img.shape
            img=img.repeat(3).reshape(H,W,3)
        if img.shape[2]==4:
            img=color.rgba2rgb(img)
        H,W,C=img.shape
        scale1,scale2=self.rows/H,self.cols/W
        scale=min(scale1,scale2)
        destImage=np.zeros((self.rows,self.cols,C),dtype=np.float32)
        ystart=int((self.rows-scale*H)/2);yend=ystart+int(scale*H)
        xstart=int((self.cols-scale*W)/2);xend=xstart+int(scale*W)
        img=transform.resize(img,(int(H*scale),int(W*scale),C),mode="constant",anti_aliasing=True)
        destImage[ystart:yend,xstart:xend]=(img-np.array(cfg.pix_mean))/cfg.std
        destImage=destImage.transpose((2,0,1))
        image=torch.from_numpy(destImage)
        return image,gt_label
        return None

class trainer(object):
    def __init__(self,trainLoader,testLoader,model,epoch=100,eps=1e-3,savePath="./"):
        self.trainLoader=trainLoader
        self.testLoader=testLoader
        self.model=model
        # self.optimizer=torch.optim.SGD(self.model.parameters(),lr=0.01, momentum=0.9)
        self.optimizer=torch.optim.Adam(self.model.parameters())
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode="min",patience=3,threshold=0.0001)
        self.start_index=1
        self.epoch=epoch
        self.eps=eps
        self.vis=visdom.Visdom(env="imageNet")
        self.interval=1
        self.checker=EarlyStopping(delta=self.eps)
        self.device=device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.currentId=self.start_index
        self.savePath=savePath
    
    def train(self,index):
        self.model.train()
        sum=0.
        for i,(imgData,gt_label) in enumerate(self.trainLoader):
            imgData=imgData.to(self.device)
            gt_label=gt_label.to(self.device)
            self.optimizer.zero_grad()
            x=self.model(imgData)
            loss=self.model.calcuLoss(x,gt_label)
            loss.backward()
            self.optimizer.step()
            sum+=loss.item()
            self.vis.line(Y=[loss.item()/64],X=[i],win="train_{}".format(index),name="train",update="append",opts=dict(showlegend=True))
        self.vis.line(Y=[sum/(len(self.trainLoader)*64)],X=[index],win="loss",name="train",update="append",opts=dict(showlegend=True))
    
    def test(self,index):
        with torch.no_grad():
            self.model.eval()
            matrix=cfsm(1000)
            sum=0.
            for i,(imgData,gt_label) in enumerate(self.testLoader):
                imgData=imgData.to(self.device)
                label=gt_label.to(self.device)
                x=self.model(imgData)
                loss=self.model.calcuLoss(x,label)
                x=F.softmax(x,dim=1)
                maxIndex=x.argmax(dim=1)
                matrix.setvalue(maxIndex.cpu(),gt_label)
                sum+=loss.item()
            average_loss=sum/(len(self.testLoader)*64)
            self.vis.line(Y=[average_loss],X=[index],win="loss",name="val",update="append",opts=dict(showlegend=True))
            cfsmImg=matrix.show()
            self.vis.image(cfsmImg,win="confusion Matrix({})".format(index),opts=dict(title="confusion Matrix"))
            self.checker(average_loss,self)

    def __call__(self):
        for index in range(self.start_index,self.epoch):
            self.train(index)
            self.currentId=index
            if(index%self.interval==0):
                self.test(index)
            if self.checker.early_stop:
                print("early stop! {}".format(self.checker.best_score))
                break
            
    def save(self,index=None):
        if index is None:
            index=self.currentId
        torch.save({"epoch":index,"state_dict":self.model.state_dict(),\
            "optimizer":self.optimizer.state_dict()},"{}SNet146_{}.pth.tar".format(self.savePath,index))
        print("{}model{}.pth.tar saved".format(self.savePath,index))

    def load(self,modeDir):
        checkpoint=torch.load(modeDir)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_index=checkpoint["epoch"]+1

    def continus_train(self,modelPath):
        self.load(modeDir)
        self.__call__()

    def set_lr(self,factor=0.1):
        for lr_param in self.optimizer.param_groups:
            lr_param["lr"]*=factor
    
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader=Data.DataLoader(dataset("/home/huangeryu/Data/imagenet/train","train.txt"),batch_size=64,shuffle=True,num_workers=4)
    val_loader=Data.DataLoader(dataset("/home/huangeryu/Data/imagenet/val","val.txt"),batch_size=64,num_workers=4)
    shufflenet=Shufflev2Net(132,stage5=False)
    net=SNet146(shufflenet).to(device)
    # pdb.set_trace()
    solver=trainer(train_loader,val_loader,net)
    solver()


def testImage():
    trainset=dataset("/home/huangeryu/Data/imagenet/train","train.txt")
    testset=dataset("/home/huangeryu/Data/imagenet/val","val.txt")
    for data in trainset:
        pass
    print("-------------------------------")
    for data in testset:
        pass

if __name__=="__main__":
    # warnings.filterwarnings("error",category=UserWarning)
    main()
    # testImage()

