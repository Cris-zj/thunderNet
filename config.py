import math
def GET_ANCHOR():
    scales=[32,64,96,128,256,320,512]
    ratios=[(1,2),(2,3),(3,4),(1,1),(4,3),(3,2),(2,1)]
    anchor=list()
    for item in scales:
        aspect=item*item
        for w,h in ratios:
            per=math.sqrt(aspect/(w*h))
            x=int(w*per)
            y=int(h*per)
            anchor.append([x,y])
    # print(anchor)
    return anchor

class Config:
    #输入图片大小
    input_width=480
    input_hight=480

    #选择用于RPN训练的iou阈值
    neg_rpn_threshold=0.3
    pos_rpn_threshold=0.7
    
    #选择正负样本的数量用于rpn训练
    num_rpn_loss=256
    pos_rpn_loss=128
    neg_rpn_loss=128

    #刷选建议框时的nms阈值
    propose_nms_threshold=0.7

    #刷选正负roi时的iou阈值
    neg_roi_threshold=0.1
    pos_roi_threshold=0.5

    #用于检测子网络的正负样本数量
    num_subnet_loss=128
    pos_subnet_loss=64
    neg_subnet_loss=64

    num_subnet_test=20

    #rpn阶段选择的建议框用于nms
    num_propose=2000
    num_propose_test=600

    #经过nms后，需要选择用于subnet的建议框,
    num_propose_roi=512

    #原始图片到特征的下采用数以及特征的大小
    feature_strid=16
    feature_size=[30,30]
    feature_roi_size=7

    #featuremap中每个anchor对应的通道数量
    alpha=5
    
    #每个roi的数量（alpha×feature_size×feature_size）
    thin_channels=alpha*feature_roi_size*feature_roi_size

    #anchor
    anchor=GET_ANCHOR()

    #用于检测物体的最宽度
    ignore_width=16

    #训练的最大epoch
    epoch=20

    #梯度更新的batch
    grad_step=1

    #测试间隔
    test_interval=1

    #保存的间隔
    saved_interval=1

    #学习率设置
    lr=0.1
    lr_step=10
    momentum=0.9
    gamma=1
    weight_decay=0.00001

    #rpn和subnet损失的权重
    rpn_cls_weight=1
    rpn_loc_weight=1
    subnet_cls_weight=1
    subnet_loc_weight=1

    #数据相关
    database_class_num=21
    pix_mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    #权重初始化
    mean=0.0
    std=0.01
    bias=0.0

    #fine-tuning参数设置
    finetuning=False
    keywords=["sam","subnet"]
    pre_lr=0.00001
    new_lr=0.1