import torchvision.models as models
import torch.nn as nn
import torch as t
import numpy as np
from torch.nn import functional as F


class build_model(nn.Module):
    '''
    extract feature
    '''

    def __init__(self):
        super(build_model, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.div_2 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
          )
        self.div_4 = nn.Sequential(resnet.maxpool,resnet.layer1)
        self.div_8 = nn.Sequential(resnet.layer2)
        self.div_16 = nn.Sequential(resnet.layer3)

    def forward(self, x):
        feature = []

        first_feature = self.div_2(x)
        second_feature = self.div_4(first_feature)
        third_feature = self.div_8(second_feature)
        fourth_feature = self.div_16(third_feature)
        feature.extend([first_feature, second_feature, third_feature, fourth_feature])
        return feature


def base_anchor(feature_size, img_size, anchor_size, ratios = [0.5, 1, 2]):
    import math
    '''
    size : (n)
    ratio:(a,b,c)
    '''

    base_size = 8
    py = base_size / 2.
    px = base_size / 2.
    anchor = np.zeros((len(ratios), 4), dtype= np.float32)
    for index, ratio in enumerate(ratios):

        high = base_size * anchor_size * np.sqrt(ratios[index])
        width = base_size * anchor_size  * np.sqrt(1. / ratios[index])

        anchor[index] = [py-high/2, px-width/2, py + high/2, px + width/2]

    y_stride = img_size[0] // feature_size[0]
    x_stride = img_size[1] // feature_size[1]
    y_ = np.arange(0, img_size[0], y_stride)
    x_ = np.arange(0 , img_size[1], x_stride)
    x_, y_ = np.meshgrid(x_, y_)
    point_stride = np.stack((y_.flatten(), x_.flatten(), y_.flatten(), x_.flatten()), axis = 1)
    all_anchor = anchor + point_stride[:, None, :]
    all_anchor = all_anchor.reshape(-1, 4)

    return all_anchor


class RPN_net(nn.Module):

    def __init__(self,  size = [2, 4, 8, 16, 32]):
        super(RPN_net, self).__init__()
        n_anchor = 3

        self.p5_conv1 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.p5_score = nn.Conv2d(1024, n_anchor * 2, 1, 1, 0)
        self.p5_loc = nn.Conv2d(1024, n_anchor * 4, 1, 1, 0)
        self.p5_conv1_1 =  nn.Conv2d(1024, 512, 3, 1, 1)

        self.p4_conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.p4_score = nn.Conv2d(512, n_anchor * 2, 1, 1, 0)
        self.p4_loc = nn.Conv2d(512, n_anchor * 4, 1, 1, 0)
        self.p4_conv1_1 = nn.Conv2d(512, 256, 3, 1, 1)

        self.p3_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.p3_score = nn.Conv2d(256, n_anchor * 2, 1, 1, 0)
        self.p3_loc = nn.Conv2d(256, n_anchor * 4, 1, 1, 0)
        self.p3_conv1_1 = nn.Conv2d(256, 64, 3, 1, 1)

        self.p2_conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.p2_score = nn.Conv2d(64, n_anchor * 2, 1, 1, 0)
        self.p2_loc = nn.Conv2d(64, n_anchor * 4, 1, 1, 0)

    def forward(self, features, img_size , scale = 1.):
        # '''
        # x: list(feature_4, feature_8, feature_16)
        # '''
        feature_2, feature_4, feature_8, feature_16 = features

        p5_cls = self.p5_loc(self.p5_conv1(feature_16))
        p5_score = self.p5_score(self.p5_conv1(feature_16))
        p5_rpn_cls, p5_rpn_score, p5_rpn_fg_score = self.generate_loc_score(feature_16,
                                                                            p5_cls,p5_score)
        p5_anchors = base_anchor(feature_16.shape[-2:], img_size, 32, ratios=[0.5, 1, 2])
        # print(p5_anchors.shape, p5_rpn_cls.shape)

        y = F.interpolate(feature_16, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.p5_conv1_1(y)
        feature_8 = t.add(feature_8, y)
        p4_cls = self.p4_loc(self.p4_conv1(feature_8))
        p4_score = self.p4_score(self.p4_conv1(feature_8))
        p4_rpn_cls, p4_rpn_score, p4_rpn_fg_score = self.generate_loc_score(feature_8,
                                                                            p4_cls, p4_score)
        p4_anchors = base_anchor(feature_8.shape[-2:], img_size, 16, ratios=[0.5, 1, 2])
        # print(p4_anchors.shape, p4_rpn_cls.shape)

        y = F.interpolate(feature_8, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.p4_conv1_1(y)
        feature_4 = t.add(feature_4, y)
        p3_cls = self.p3_loc(self.p3_conv1(feature_4))
        p3_score = self.p3_score(self.p3_conv1(feature_4))
        p3_rpn_cls, p3_rpn_score, p3_rpn_fg_score = self.generate_loc_score(feature_4,
                                                                            p3_cls, p3_score)
        p3_anchors = base_anchor(feature_4.shape[-2:], img_size, 8, ratios=[0.5, 1, 2])
        # print(p3_anchors.shape, p3_rpn_cls.shape)


        y = F.interpolate(feature_4, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.p3_conv1_1(y)
        feature_2 = t.add(feature_2, y)
        p2_cls = self.p2_loc(self.p2_conv1(feature_2))
        p2_score = self.p2_score(self.p2_conv1(feature_2))
        p2_rpn_cls, p2_rpn_score, p2_rpn_fg_score = self.generate_loc_score(feature_2,
                                                                            p2_cls, p2_score)
        p2_anchors = base_anchor(feature_2.shape[-2:], img_size, 4, ratios=[0.5, 1, 2])
        # print(p2_anchors.shape, p2_rpn_cls.shape)

        rpn_locs = [p5_rpn_cls, p4_rpn_cls, p3_rpn_cls, p2_rpn_cls]
        rpn_scores = [p5_rpn_score, p4_rpn_score, p3_rpn_score, p2_rpn_score]
        rpn_fg_scores = [p5_rpn_fg_score, p5_rpn_fg_score, p5_rpn_fg_score, p5_rpn_fg_score]
        anchors = [p5_anchors, p4_anchors, p3_anchors, p2_anchors ]

        rpn_locs = t.cat(rpn_locs, 1)
        rpn_scores = t.cat(rpn_scores, 1)
        rpn_fg_scores = t.cat(rpn_fg_scores, 1)
        anchors = np.concatenate(anchors, axis = 0)

        return rpn_locs, rpn_scores, rpn_fg_scores, anchors

    def generate_loc_score(self, feature, rpn_loc, rpn_score):
        n, _, hh, ww = feature.shape

        rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_score = F.softmax(rpn_score, dim=2)
        rpn_fg_score = rpn_softmax_score[..., 1].contiguous()
        rpn_fg_score = rpn_fg_score.view(n, -1)

        return (rpn_loc, rpn_score, rpn_fg_score)

def bbox_iou( bbox, anchor):
    '''
    bbox: n * 4
    anchor: m * 4
    '''

    area_bbox = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:,1])
    area_anchor = (anchor[:, 2] - anchor[:, 0]) * (anchor[:, 3] - anchor[:, 1])

    bbox = bbox[: , None, :]

    y_1 = np.maximum(bbox[..., 0], anchor[..., 0])
    x_1 = np.maximum(bbox[..., 1], anchor[..., 1])
    y_2 = np.minimum(bbox[..., 2], anchor[..., 2])
    x_2 = np.minimum(bbox[..., 3], anchor[..., 3])

    heigh = np.maximum(y_2-y_1, 0)
    width = np.maximum(x_2-x_1, 0)

    iner_area = heigh * width

    iou = iner_area / (area_bbox[:, None] + area_anchor[None, :] - iner_area)

    return iou


class ProposalTargetCreator:
    def __init__(self, n_sample=256,
                 pos_iou_thresh = 0.7, neg_iou_thresh = 0.3,
                 pos_ratio = 0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bboxs, anchors, img_size):
        '''
        img_size: (h,w)
        '''
        h, w = img_size
        num_bboxs = bboxs.shape[0]

        inside_index = np.where((bboxs[:, 0] >= 0) & (bboxs[:, 1] >= 0) &
                                (bboxs[:, 2] <= h) & (bboxs[:, 3] <= w))[0]

        inside_anchor = anchors[inside_index]
        ious = bbox_iou(bboxs, inside_anchor)

        positive_max_index = []
        max_ious = np.max(ious, axis=1)
        max_ious_sign = np.argmax(ious, axis=1)

        filter_positive = np.where(max_ious > self.pos_iou_thresh)[0]
        filter_iou_row = np.max(ious, axis=0)
        positive_max_index.extend(list(filter_positive))
        positive_max_index.extend(list(filter_iou_row))
        positive_max_index = list(set(positive_max_index))

        positive_anchor = inside_anchor[positive_max_index]
        positive_box = bboxs[max_ious_sign[positive_max_index]]
        positive_index = inside_index[positive_max_index]

        negative_index = np.where(max_ious < self.neg_iou_thresh)[0]
        negative_index = inside_index[negative_index]
















bbox = np.array([[0,0,1,1],[1,1,2,2]])
anchor = np.array([[0,0,3,3],[1,1,2,2],[1,1,3,3]])
iou = bbox_iou(bbox, anchor)
print(iou)






# a = t.Tensor(1,3,112, 112)
# model_1 = build_model()
# rpn = RPN_net()
# features = model_1.forward(a)
# result = rpn.forward(features, (112, 112))



# print(result[0].shape, result[3].shape)
# anchor = base_anchor(8, [0.5, 1, 2])
# print(anchor)