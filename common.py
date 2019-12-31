import numpy as np
import torch
import torchvision.ops as ops
import torch.nn.functional as F
import pdb
from matplotlib import pyplot as plt

def gen_anchor(anchor,featureSize,strid)->np.ndarray:
    offset=strid//2
    srcH,srcW=featureSize[0]*strid,featureSize[1]*strid
    center=[[x*strid+offset,y*strid+offset] for y in range(featureSize[0]) for x in range(featureSize[1])]
    center=np.array(center).reshape(-1,1,2)
    anchor=np.array(anchor,dtype=float).reshape(-1,2)//2
    lt=center-anchor
    rd=center+anchor
    baseAnchor=np.concatenate((lt,rd),axis=2).reshape(-1,4)
    validIndex=np.where( ((baseAnchor[:,:2]>=0) & (baseAnchor[:,2:]<(srcW,srcH))).all(axis=1) )[0]
    # validIndex=np.where(((baseAnchor[:,:2]>-1000) & (baseAnchor[:,2:]<10000)).any(axis=1))[0]
    return baseAnchor,validIndex

def iou3d(box_a,box_b):
    #box_a.shape=(n,r1,4) box_b.shape=(r2,4)
    #return (n,r1,r2)
    if(type(box_b)!=torch.Tensor):
        box_b=torch.from_numpy(box_b)
    lt=np.maximum(box_a[:,:,None,0:2],box_b[:,0:2])
    rd=np.minimum(box_a[:,:,None,2:4],box_b[:,2:4])
    area_a=torch.prod(box_a[:,:,2:]-box_a[:,:,0:2],axis=2).unsqueeze(dim=-1)
    area_b=torch.prod(box_b[:,2:]-box_b[:,0:2],axis=1)
    area=torch.prod(rd-lt,axis=3)*(lt<rd).all(axis=3)
    return area /(area_a+area_b-area)

def iou2d(box_a,box_b):
    #box_a.shape=(R1,4),box_b=(R2,4)
    #return (R1,R2)
    if(type(box_b)!=torch.Tensor):
        box_b=torch.from_numpy(box_b)
    lt=np.maximum(box_a[:,None,0:2],box_b[:,0:2])
    rd=np.minimum(box_a[:,None,2:4],box_b[:,2:4])
    area_a=torch.prod(box_a[:,2:]-box_a[:,0:2],axis=1).unsqueeze(dim=-1)
    area_b=torch.prod(box_b[:,2:]-box_b[:,0:2],axis=1)
    area=torch.prod(rd-lt,axis=2)*(lt<rd).all(axis=2)
    return area /(area_a+area_b-area)

def loc2box2d(predict,anchorbox):
    #predict.shape=(r1,4),anchorbox.shape=(r1,4)
    if(type(anchorbox)!=torch.Tensor):
        anchorbox=torch.from_numpy(anchorbox)
    wh=anchorbox[:,2:]-anchorbox[:,:2]
    center=anchorbox[:,:2]+wh/2
    ret=torch.zeros_like(predict,dtype=torch.float)
    new_center=wh*predict[:,:2]+center
    new_wh=torch.exp(predict[:,2:])*wh
    ret[:,:2]=new_center-new_wh/2
    ret[:,2:]=new_center+new_wh/2
    return ret

def loc2box(predict,anchorbox):
    #predict.shape=(n,r1,4),anchorbox.shape=(n,r1,4)
    if(type(anchorbox)!=torch.Tensor):
        anchorbox=torch.from_numpy(anchorbox)
    wh=anchorbox[:,:,2:]-anchorbox[:,:,:2]
    center=anchorbox[:,:,:2]+wh/2
    ret=torch.zeros_like(predict,dtype=torch.int)
    new_center=wh*predict[:,:,:2]+center
    new_wh=torch.exp(predict[:,:,2:])*wh
    ret[:,:,:2]=new_center-new_wh/2
    ret[:,:,2:]=new_center+new_wh/2
    return ret

def box2loc(predict,anchorbox):
    if(type(anchorbox)!=torch.Tensor):
        anchorbox=torch.from_numpy(anchorbox)
    wh=anchorbox[:,:,2:]-anchorbox[:,:,:2]
    center=anchorbox[:,:,:2]+wh/2
    pre_wh=predict[:,:,2:]-predict[:,:,:2]
    pre_center=predict[:,:,:2]+pre_wh/2
    ret=torch.zeros_like(predict,dtype=torch.float)
    ret[:,:,:2]=(pre_center-center)/wh
    ret[:,:,2:]=np.log(pre_wh/wh)
    return ret

def choose_box_for_rpn_loss(gt_box,anchorbox,validIndex,length,num,neg_threshold=0.3,pos_threshold=0.7)->np.ndarray:
    assert(length%anchorbox.shape[0]==0)
    if(gt_box.ndim==2):gt_box.unqueeze(0)
    batch_size=length//anchorbox.shape[0]
    anchorbox=anchorbox[validIndex]
    box_iou=iou3d(gt_box,anchorbox)
    table=np.full(length,-1,dtype=np.int).reshape((batch_size,-1))
    validtable=table[:,validIndex]
    maxindex=box_iou.argmax(axis=2)
    for i,sub_box_iou in enumerate(box_iou):
        pos_threshold_index=np.where(sub_box_iou>pos_threshold)
        pos_num=len(pos_threshold_index[0])
        if(pos_num>num//2):
            random_index=np.random.randint(0,pos_num,size=num//2)
            validtable[i,pos_threshold_index[1][random_index]]=pos_threshold_index[0][random_index]+1
            neg_num=num//2
        else:
            validtable[i,pos_threshold_index[1]]=pos_threshold_index[0]+1
            # print(pos_threshold_index[0]+1)
            neg_num=num-pos_num
        print("rpn_pos_num={}".format(pos_num))
        neg_index=np.where((sub_box_iou<neg_threshold).all(axis=0))[0]
        neg_index=np.random.choice(neg_index,size=neg_num)
        validtable[i,neg_index]=0
        validtable[i,maxindex[i]]=np.arange(len(maxindex[i]))+1
        table[i:,validIndex]=validtable
    return table

def choose_roi_for_subnet_only_forward(rpn_cls,rpn_loc,propose_num,propose_roi_num,anchorbox,
                        validIndex,nms_threshold=0.7,rows=320,cols=320,width_threshold=16):
    #type(rpn_loc)==torch.Tensor,type(rpn_cls)==torch.Tensor
    rpn_cls=rpn_cls.cpu()
    rpn_loc=rpn_loc.cpu()
    k,h,w=rpn_cls.shape[1]//2,rpn_cls.shape[2],rpn_cls.shape[3]
    batch=rpn_cls.shape[0]
    rpn_cls=F.softmax(rpn_cls.reshape(-1,k,2,h*w),dim=2).permute(0,3,1,2).reshape(-1,h*w,k,2)
    rpn_loc=rpn_loc.reshape(-1,k,4,h*w).permute(0,3,1,2).reshape(-1,h*w*k,4)
    pos_scores=rpn_cls[:,:,:,1].reshape(batch,-1)
    pos_scores=pos_scores[:,validIndex]
    scores_value,pos_scores_index=pos_scores.sort(descending=True,dim=1)
    #选择概率最大的propose_num个建议框
    scores_value=scores_value[:,:propose_num]
    pos_scores_index=pos_scores_index[:,:propose_num]
    valid_loc=rpn_loc[:,validIndex]
    batch_index=np.arange(batch).reshape(batch,1)
    propose_loc=valid_loc[batch_index,pos_scores_index]
    anchorbox=anchorbox[validIndex][pos_scores_index]
    src_box=loc2box(propose_loc,anchorbox)
    #对建议框进行裁剪
    lt=np.maximum(src_box[:,:,:2],[0,0])
    src_box[:,:,:2]=lt
    rd=np.minimum(src_box[:,:,2:],[cols-1,rows-1])
    src_box[:,:,2:]=rd
    filted_indices=(rd-lt>width_threshold).all(axis=2)
    tmp=[]
    for i in range(len(src_box)):
        box,score,index = src_box[i],scores_value[i],filted_indices[i]
        validbox=box[index].to(torch.float32)
        validscore=score[index]
        #执行极大值抑制，选择建议框
        indices=ops.nms(validbox,validscore,nms_threshold)[:propose_roi_num]
        subbox=validbox[indices]
        tmp.append(subbox)
    rois=torch.stack(tmp)
    if(rois.ndim==2): rois=rois.unsqueeze(dim=0)
    return rois

def showImg(img,boxes,gt_box):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img,)
    if(boxes.ndim==1):boxes=boxes.reshape(-1,4)
    for i in range(len(boxes)):
        box=boxes[i]
        rect=plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False, edgecolor = 'red',linewidth=1)
        ax.add_patch(rect)
    for i in range(len(gt_box)):
        box=gt_box[i]
        rect=plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False, edgecolor = 'g',linewidth=1)
        txt=ax.text(box[0]+10,box[1]+10,"{}".format(i),fontsize=12,color="g")
        ax.add_patch(rect)
    plt.show()

def choose_roi_for_subnet(gt_box,rpn_cls,rpn_loc,propose_num,propose_roi_num,anchorbox,validIndex,
                        nms_threshold=0.7,iou_threshold=0.5,pos_num=32,neg_num=96,rows=320,cols=320,width_threshold=16):
    #type(rpn_loc)==torch.Tensor,type(rpn_cls)==torch.Tensor
    rpn_cls=rpn_cls.cpu()
    rpn_loc=rpn_loc.cpu()
    k,h,w=rpn_cls.shape[1]//2,rpn_cls.shape[2],rpn_cls.shape[3]
    batch=rpn_cls.shape[0]
    rpn_cls=F.softmax(rpn_cls.reshape(-1,k,2,h*w),dim=2).permute(0,3,1,2).reshape(-1,h*w,k,2)
    rpn_loc=rpn_loc.reshape(-1,k,4,h*w).permute(0,3,1,2).reshape(-1,h*w*k,4)
    pos_scores=rpn_cls[:,:,:,1].reshape(batch,-1)
    pos_scores=pos_scores[:,validIndex]
    scores_value,pos_scores_index=pos_scores.sort(descending=True,dim=1)
    #选择概率最大的propose_num个建议框
    scores_value=scores_value[:,:propose_num]
    pos_scores_index=pos_scores_index[:,:propose_num]
    valid_loc=rpn_loc[:,validIndex]
    batch_index=np.arange(batch).reshape(batch,1)
    propose_loc=valid_loc[batch_index,pos_scores_index]
    anchorbox=anchorbox[validIndex][pos_scores_index]
    src_box=loc2box(propose_loc,anchorbox)
    #对建议框进行裁剪
    lt=np.maximum(src_box[:,:,:2],[0,0])
    src_box[:,:,:2]=lt
    rd=np.minimum(src_box[:,:,2:],[cols-1,rows-1])
    src_box[:,:,2:]=rd
    filted_indices=(rd-lt>width_threshold).all(axis=2)
    tmp=[];gt_label=[]
    for i in range(len(gt_box)):
        gtbox,box,score,index = gt_box[i],src_box[i],scores_value[i],filted_indices[i]
        validbox=box[index].to(torch.float32)
        validscore=score[index]
        #-----------------------------
        # img=np.zeros((480,480,3),dtype=float)
        # print(len(validbox))
        #执行极大值抑制，选择建议框
        indices=ops.nms(validbox,validscore,nms_threshold)[:propose_roi_num]
        subbox=validbox[indices]
        # print(len(subbox))
        box_iou=iou2d(gtbox,subbox)
        pos_threshold_index=np.where(box_iou>iou_threshold)
        length=len(pos_threshold_index[0])
        # print(pos_threshold_index)
        print("subnet_pos_num={}".format(length))
        if length>pos_num:
            random_index=np.random.randint(0,length,size=pos_num)
            pos_indices=pos_threshold_index[1][random_index]
            pos_label=pos_threshold_index[0][random_index]
            length=pos_num
        else:
            pos_indices=pos_threshold_index[1]
            pos_label=pos_threshold_index[0]
        # print(pos_label)
        # showImg(img,subbox[pos_indices],gtbox)
        # pdb.set_trace()
        neg_threshold_index=np.where((box_iou<0.1).all(axis=0))
        if(len(neg_threshold_index[0])==0):
            length=0
            neg_indices=[]
            print("sub_neg_num=0")
        else:
            length=pos_num+neg_num-length
            neg_indices=np.random.choice(neg_threshold_index[0],size=length)
        neg_label=np.full(length,-1)
        selected_box=subbox[np.concatenate((pos_indices,neg_indices))]
        label=np.concatenate((pos_label,neg_label))
        label=torch.from_numpy(label)
        tmp.append(selected_box)
        gt_label.append(label)
    rois=torch.stack(tmp)
    if(rois.ndim==2): rois=rois.unsqueeze(dim=0)
    labels=torch.stack(gt_label)
    if(labels.ndim==1): labels=labels.unsqueeze(dim=0)
    return rois,labels

def warp_rois_for_psroi(rois,strid=16):
    if(rois.dtype==torch.float32):
        rois=rois.to(torch.int)
    if rois.ndim==2:
        rois=rois.unsqueeze(dim=0)
    rois[:,:,2:]=rois[:,:,2:]-rois[:,:,:2]+1
    rois=rois//strid
    return rois

def box2loc2d(predict,anchorbox):
    if(type(anchorbox)!=torch.Tensor):
        anchorbox=torch.from_numpy(anchorbox)
    wh=anchorbox[:,2:]-anchorbox[:,:2]
    center=anchorbox[:,:2]+wh/2
    pre_wh=predict[:,2:]-predict[:,:2]
    pre_center=predict[:,:2]+pre_wh/2
    ret=torch.zeros_like(predict,dtype=torch.float)
    ret[:,:2]=(pre_center-center)/wh
    ret[:,2:]=np.log(pre_wh/wh)
    return ret

def clc_rpn_loss(rpn_cls,rpn_loc,table,gt_box,anchor_box):
    device=rpn_cls.device
    batch,channel,h,w=rpn_cls.shape
    k=channel//2
    filted_rpn_cls=rpn_cls.reshape(batch,k,2,h*w).permute(0,3,1,2).reshape(batch*h*w*k,2)
    cls_label=np.minimum(table,1).reshape(-1)
    cls_label=torch.from_numpy(cls_label).to(device=device)
    cls_loss=F.cross_entropy(filted_rpn_cls,cls_label,ignore_index=-1,reduction="mean")
    batch_index=np.arange(batch).reshape(batch,1)
    filted_rpn_loc=rpn_loc.reshape(batch,k,4,h*w).permute(0,3,1,2).reshape(batch,h*w*k,4)
    indices=np.where(table>0)
    pos_table=table[indices]-1
    gt_box=gt_box[indices[0],pos_table]
    gt_anchor=anchor_box[indices[1]]
    filted_rpn_loc=filted_rpn_loc[indices].reshape(-1,4)
    gt_loc=box2loc2d(gt_box,gt_anchor).to(device=device)
    loc_loss=F.smooth_l1_loss(filted_rpn_loc,gt_loc,reduction="mean")
    # print("rpn_cls_loss={} rpn_loc_loss={}".format(cls_loss,loc_loss))
    return cls_loss,loc_loss

def clc_subnet_loss(sub_cls,sub_reg,roi_labels,roi_box,gt_box,gt_label,strid=16):
    batch=roi_box.size(0)
    device=sub_cls.device
    indices=np.where(roi_labels>-1)
    cls_label=torch.zeros_like(roi_labels,dtype=torch.int64)
    cls_label[indices]=gt_label[indices[0],roi_labels[indices]]
    # print(cls_label[indices])
    cls_label=cls_label.reshape(-1).to(device=device)
    cls_loss=F.cross_entropy(sub_cls,cls_label,reduction="mean",)
    pos_reg_label=roi_labels[indices]
    reg_label=gt_box[indices[0],pos_reg_label].reshape(-1,4)
    # print(gt_label)
    roi_reg_box=roi_box[indices].reshape(-1,4)
    temp_label=roi_labels.reshape(-1)
    reg_indices=np.where(temp_label>-1)
    # pdb.set_trace()
    if(len(reg_indices[0])>0):
        valid_reg=sub_reg[reg_indices[0]]
        gt_reg=box2loc2d(reg_label,roi_reg_box).to(device=device)
        reg_loss=F.smooth_l1_loss(valid_reg,gt_reg,reduction="mean")
        # print(valid_reg)
        # print(gt_reg)
        # print("subnet_cls_loss={} subnet_reg_loss={}".format(cls_loss,reg_loss))
        return cls_loss,reg_loss
    else:
        # print("subnet_cls_loss={} subnet_reg_loss={}".format(cls_loss,0.0))
        return cls_loss,None
    
    
    
