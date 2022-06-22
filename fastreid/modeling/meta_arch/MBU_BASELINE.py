# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import copy
import numpy as np
from torchvision import transforms
from numpy import float32
import torch
from torch import nn
import pdb
import torchvision
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.backbones.resnet import Bottleneck



from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.model_utils import weights_init_kaiming
from fastreid.layers import GeneralizedMeanPoolingP, Flatten
from fastreid.modeling.meta_arch.STN import *
from MBU_reid.build_losses import reid_losses, map_losses0, map_losses1,map_losses2,map_losses3,map_losses4,map_losses5
from MBU_reid.blackhead import BlackHead
from fastreid.modeling.models.models_utils.rga_modules import RGA_Module
from fastreid.modeling.models.model_keypoints import ScoremapComputer, compute_local_features
from fastreid.modeling.models.model_gcn import GraphConvNet, generate_adj

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BottleneckA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckA, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BottleneckB(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckB, self).__init__()
        assert inplanes == (planes * 4), 'inplanes != planes * 4'
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class BNClassifiers(nn.Module):

    def __init__(self, in_dim, class_num, branch_num):
        super(BNClassifiers, self).__init__()
        self.in_dim = in_dim
        self.class_num = class_num
        self.branch_num = branch_num
        for i in range(self.branch_num):
            setattr(self, 'classifier_{}'.format(i), BNClassifier(self.in_dim, self.class_num))

    def __call__(self, feature_vector_list):
        assert len(feature_vector_list) == self.branch_num
        bned_feature_vector_list, cls_score_list = [], []
        for i in range(self.branch_num):
            feature_vector_i = feature_vector_list[i]
            classifier_i = getattr(self, 'classifier_{}'.format(i))
            bned_feature_vector_i, cls_score_i = classifier_i(feature_vector_i)
            bned_feature_vector_list.append(bned_feature_vector_i)
            cls_score_list.append(cls_score_i)
        return bned_feature_vector_list, cls_score_list


@META_ARCH_REGISTRY.register()
class MBU_BASELINE(nn.Module):
    def __init__(self, cfg, spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8, height=256, width=192):
        super().__init__()
        self._cfg = cfg
        resnet = build_backbone(cfg)
        self.num_stripes = 4
        self.num_classes = 632
        self.local_4_conv_list = nn.ModuleList()
        self.rest_4_conv_list = nn.ModuleList()
        self.relation_4_conv_list = nn.ModuleList()
        self.global_4_max_conv_list = nn.ModuleList()
        self.global_4_rest_conv_list = nn.ModuleList()
        self.global_4_pooling_conv_list = nn.ModuleList()
        for i in range(4):
            self.local_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)))
        for i in range(4):
            self.rest_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)))
        self.global_4_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)))
        self.global_4_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)))
        for i in range(4):
            self.relation_4_conv_list.append(nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)))
        self.global_4_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(256 * 2, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)))
        if self.num_classes > 0:
            self.fc_local_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(256, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_4_list.append(fc)
            self.fc_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(256, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_4_list.append(fc)
            self.fc_local_rest_6_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(256, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_6_list.append(fc)
            self.fc_local_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(256, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_4_list.append(fc)
            self.fc_global_4_list = nn.ModuleList()
            fc = nn.Linear(256, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_4_list.append(fc)
            self.fc_global_max_4_list = nn.ModuleList()
            fc = nn.Linear(256, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_4_list.append(fc)
            self.fc_global_rest_4_list = nn.ModuleList()
            fc = nn.Linear(256, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_4_list.append(fc)

            self.seblock1 = SELayer(channel=256, reduction=16)
            self.seblock2 = SELayer(channel=512, reduction=16)


        self.device = torch.device('cuda')
        self.rga_att1 = RGA_Module(256, (height // 4) * (width // 4), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att2 = RGA_Module(512, (height // 8) * (width // 8), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)


        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            self.seblock1,
            #self.rga_att1,
            resnet.layer2,
            self.seblock2,
            #self.rga_att2,
            resnet.layer3[0],
        )
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_g_conv5 = nn.Sequential(resnet.layer4)
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            BottleneckB(2048, 512),
            BottleneckA(2048, 512),
            BottleneckA(2048, 512),)
        self.bone = nn.Sequential(copy.deepcopy(self.backbone), copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))

        if cfg.MODEL.HEADS.POOL_LAYER == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        else:
            pool_layer = nn.Identity()

        self.bone_pool = self._build_pool_reduce(copy.deepcopy(pool_layer), 2048, 1536)
        self.heads = build_reid_heads(cfg, 1280, nn.Identity())

        self.classifier_black = BlackHead(cfg, 2, 1536, nn.Identity())
        self.prob = BlackHead(cfg, 2, 2, nn.Identity())
        self.final_head = build_reid_heads(cfg, 92960, nn.Identity())
        self.heatpool = self._build_pool_reduce(copy.deepcopy(pool_layer), 2048, 512)
        self.stn = SpatialTransformBlock(1, 24, 512)
        model_dict = self.stn.state_dict()
        pretrained_dict = torch.load('MBU_reid/stn/model_best_stn.pth')
        pretrained_dict = {k[4:]: v for k, v in pretrained_dict.items() if k.startswith('stn')}
        self.stn.load_state_dict(pretrained_dict)
        for param in self.stn.parameters():
            param.requires_grad = False
        self.hsa = nn.Sequential(copy.deepcopy(self.backbone), copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.hsa_1_pool = self._build_pool_reduce(copy.deepcopy(pool_layer), 2048, 512)
        self.hsa_1_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_2_pool = self._build_pool_reduce(copy.deepcopy(pool_layer), 2048, 512)
        self.hsa_2_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_3_pool = self._build_pool_reduce(copy.deepcopy(pool_layer), 2048, 512)
        self.hsa_3_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_4_pool = self._build_pool_reduce(copy.deepcopy(pool_layer), 2048, 512)
        self.hsa_4_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_head = build_reid_heads(cfg, 2048, nn.Identity())


        self.han_pool_1 = copy.deepcopy(pool_layer)
        self.han_c_att_1 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                         nn.Linear(1024, 2048, bias=False))
        self.han_c_att_1[0].apply(weights_init_kaiming)
        self.han_c_att_1[1].apply(weights_init_kaiming)

        self.han_pool_2 = copy.deepcopy(pool_layer)
        self.han_c_att_2 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                         nn.Linear(1024, 2048, bias=False))
        self.han_c_att_2[0].apply(weights_init_kaiming)
        self.han_c_att_2[1].apply(weights_init_kaiming)

        self.han_pool_3 = copy.deepcopy(pool_layer)
        self.han_c_att_3 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                         nn.Linear(1024, 2048, bias=False))
        self.han_c_att_3[0].apply(weights_init_kaiming)
        self.han_c_att_3[1].apply(weights_init_kaiming)

        self.han_pool_4 = copy.deepcopy(pool_layer)
        self.han_c_att_4 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                         nn.Linear(1024, 2048, bias=False))
        self.han_c_att_4[0].apply(weights_init_kaiming)
        self.han_c_att_4[1].apply(weights_init_kaiming)

        self.scoremap_computer = ScoremapComputer(10.0).to(self.device)
        self.scoremap_computer = self.scoremap_computer.eval()

        self.bnclassifiers = BNClassifiers(2048, 632, 14)
        self.bnclassifiers2 = BNClassifiers(2048, 632, 14)


        self.linked_edges = \
            [[13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10],
             [13, 11], [13, 12],
             [0, 1], [0, 2],
             [1, 2], [1, 7], [2, 8], [7, 8], [1, 8], [2, 7],
             [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],
             ]
        self.adj = generate_adj(14, self.linked_edges, self_connect=0.0).to(self.device)
        self.gcn = GraphConvNet(self.adj, 2048, 2048, 2048, 20.0).to(self.device)

    def _build_pool_reduce(self, pool_layer, input_dim=2048, reduce_dim=256):
        pool_reduce = nn.Sequential(
            pool_layer,
            nn.Conv2d(input_dim, reduce_dim, 1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(True),
            Flatten()
        )
        pool_reduce.apply(weights_init_kaiming)
        return pool_reduce

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]
        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs["camid"]



        features = self.bone(images)
        assert (features.size(2) % self.num_stripes == 0)
        stripe_h_4 = int(features.size(2) / 4)
        local_4_feat_list = []
        final_feat_list = []
        logits_list = []
        rest_4_feat_list = []
        logits_local_rest_list = []
        logits_local_list = []
        logits_rest_list = []
        logits_global_list = []

        for i in range(4):
            local_4_feat = F.max_pool2d(
                features[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],   # 每一块是4*w  这个features是经过backbone之后得到的feature
                (stripe_h_4, features.size(-1)))
            local_4_feat_list.append(local_4_feat)
        global_4_max_feat = F.max_pool2d(features, (features.size(2), features.size(3)))
        global_4_rest_feat = (local_4_feat_list[0] + local_4_feat_list[1] + local_4_feat_list[2]
                              + local_4_feat_list[3] - global_4_max_feat) / 3
        global_4_max_feat = self.global_4_max_conv_list[0](global_4_max_feat)
        global_4_rest_feat = self.global_4_rest_conv_list[0](global_4_rest_feat)
        global_4_max_rest_feat = self.global_4_pooling_conv_list[0](torch.cat((global_4_max_feat, global_4_rest_feat), 1))
        global_4_feat = (global_4_max_feat + global_4_max_rest_feat).squeeze(3).squeeze(2)

        for i in range(4):
            rest_4_feat_list.append((local_4_feat_list[(i + 1) % 4]
                                     + local_4_feat_list[(i + 2) % 4]
                                     + local_4_feat_list[(i + 3) % 4]) / 3)
        for i in range(4):
            local_4_feat = self.local_4_conv_list[i](local_4_feat_list[i]).squeeze(3).squeeze(2)
            input_rest_4_feat = self.rest_4_conv_list[i](rest_4_feat_list[i]).squeeze(3).squeeze(2)
            input_local_rest_4_feat = torch.cat((local_4_feat, input_rest_4_feat), 1).unsqueeze(2).unsqueeze(3)
            local_rest_4_feat = self.relation_4_conv_list[i](input_local_rest_4_feat)
            local_rest_4_feat = (local_rest_4_feat
                                 + local_4_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            final_feat_list.append(local_rest_4_feat)
            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_4_list[i](local_rest_4_feat))
                logits_local_list.append(self.fc_local_4_list[i](local_4_feat))
                logits_rest_list.append(self.fc_rest_4_list[i](input_rest_4_feat))
        final_feat_list.append(global_4_feat)
        final_feat_list1 = torch.tensor([item.cpu().detach().numpy() for item in final_feat_list]).cuda()
        final_feat = final_feat_list1.permute(1,2,0)
        logits, global_feat = self.heads(final_feat, targets)

        with torch.no_grad():
            score_maps, keypoints_confidence, _ = self.scoremap_computer(images)
        feature_vector_list, keypoints_confidence = compute_local_features(
            self._cfg, features, score_maps, keypoints_confidence)
        bned_feature_vector_list, cls_score_list = self.bnclassifiers(feature_vector_list)


        gcned_feature_vector_list = self.gcn(feature_vector_list)
        bned_gcned_feature_vector_list, gcned_cls_score_list = self.bnclassifiers2(gcned_feature_vector_list)

        stn = self.stn(images)
        head = stn[0][0]
        grid_list = stn[1][0]
        head_feat = self.hsa(head)
        head_pool_feat = self.heatpool(head_feat)
        hf_1 = self.HAN_1(head_feat[:, :, 0:4, :])
        hf1_pool_feat = self.hsa_1_pool(hf_1)
        hf1_logits, hf1_pool_feat = self.hsa_1_head(hf1_pool_feat, targets)
        hf_2 = self.HAN_2(head_feat[:, :, 4:8, :])
        hf2_pool_feat = self.hsa_2_pool(hf_2)
        hf2_logits, hf2_pool_feat = self.hsa_2_head(hf2_pool_feat, targets)
        hf_3 = self.HAN_3(head_feat[:, :, 8:12, :])
        hf3_pool_feat = self.hsa_3_pool(hf_3)
        hf3_logits, hf3_pool_feat = self.hsa_3_head(hf3_pool_feat, targets)
        hf_4 = self.HAN_4(head_feat[:, :, 12:16, :])
        hf4_pool_feat = self.hsa_4_pool(hf_4)
        hf4_logits, hf4_pool_feat = self.hsa_4_head(hf4_pool_feat, targets)
        hf = torch.cat((hf1_pool_feat, hf2_pool_feat, hf3_pool_feat, hf4_pool_feat), dim=1)
        hf_logits, hf = self.hsa_head(hf, targets)
        hm1 = torch.sub(input=hf1_pool_feat, alpha=1, other=head_pool_feat)
        hm2 = torch.sub(input=hf2_pool_feat, alpha=1, other=head_pool_feat)
        hm3 = torch.sub(input=hf3_pool_feat, alpha=1, other=head_pool_feat)
        hm4 = torch.sub(input=hf4_pool_feat, alpha=1, other=head_pool_feat)
        hm = torch.cat((hm1,hm2,hm3,hm4), dim=1)
        hm_logits, hm = self.hsa_head(hm, targets)


        gf = global_feat
        cls_score_list1 = torch.tensor([item.cpu().detach().numpy() for item in cls_score_list]).cuda()
        gcned_cls_score_list1 = torch.tensor([item.cpu().detach().numpy() for item in gcned_cls_score_list]).cuda()

        pred_feat = torch.cat((gf.unsqueeze(2).repeat(1,1,14), hf.unsqueeze(2).repeat(1,1,14), hm.unsqueeze(2).repeat(1,1,14), cls_score_list1.permute(1,2,0), gcned_cls_score_list1.permute(1,2,0)), dim=1)
        pred_logits, pred_feat = self.final_head(pred_feat, targets)

        return (logits, hf1_logits, hf2_logits, hf3_logits, hf4_logits, hf_logits, pred_logits, cls_score_list,hm_logits, gcned_cls_score_list), \
               (global_feat, hf1_pool_feat, hf2_pool_feat, hf3_pool_feat, hf3_pool_feat, hf, pred_feat, feature_vector_list, hm, bned_gcned_feature_vector_list ), \
               targets, \
               grid_list, \

    def losses(self, outputs, iters=0):
        loss_dict = {}
        if iters <= int(int(self._cfg.SOLVER.MAX_ITER) * 2 / 3):  # SOLVER.MAX_ITER最大迭代次数15000
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][0], outputs[1][0], outputs[2], 0.2, 0.2, 'gf_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][1], outputs[1][1], outputs[2], 0.2, 0.2, 'hf1_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][2], outputs[1][2], outputs[2], 0.2, 0.2, 'hf2_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][3], outputs[1][3], outputs[2], 0.2, 0.2, 'hf3_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][4], outputs[1][4], outputs[2], 0.2, 0.2, 'hf4_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][5], outputs[1][5], outputs[2], 0.2, 0.2, 'hf_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][7], outputs[1][7], outputs[2], 0.2, 0.2, 'ps_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][8], outputs[1][8], outputs[2], 0.2, 0.2, 'hm_'))
            loss_dict.update(
                reid_losses(self._cfg, outputs[0][9], outputs[1][9], outputs[2], 0.2, 0.2, 'gcn_'))
            loss_dict.update(map_losses0(outputs[0][0]))
            loss_dict.update(map_losses1(outputs[0][1]))
            loss_dict.update(map_losses2(outputs[0][2]))
            loss_dict.update(map_losses3(outputs[0][3]))
            loss_dict.update(map_losses4(outputs[0][4]))
            loss_dict.update(map_losses5(outputs[0][5]))
        else:
            loss_dict.update(reid_losses(self._cfg, outputs[0][0], outputs[1][0], outputs[2], 0.143, 0.167, 'gf_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][1], outputs[1][1], outputs[2], 0.143, 0.167, 'hf1_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][2], outputs[1][2], outputs[2], 0.143, 0.167, 'hf2_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][3], outputs[1][3], outputs[2], 0.143, 0.167, 'hf3_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][4], outputs[1][4], outputs[2], 0.143, 0.167, 'hf4_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][5], outputs[1][5], outputs[2], 0.143, 0.167, 'hf_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][7], outputs[1][7], outputs[2], 0.143, 0.167, 'ps_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][8], outputs[1][8], outputs[2], 0.143, 0.167, 'hm_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][9], outputs[1][9], outputs[2], 0.143, 0.167, 'gcn_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][6], outputs[1][6], outputs[2], 0.143, 0.167, 'pred_'))
            loss_dict.update(map_losses0(outputs[0][0]))
            loss_dict.update(map_losses1(outputs[0][1]))
            loss_dict.update(map_losses2(outputs[0][2]))
            loss_dict.update(map_losses3(outputs[0][3]))
            loss_dict.update(map_losses4(outputs[0][4]))
            loss_dict.update(map_losses5(outputs[0][5]))
        return loss_dict

    def inference(self, images):
        assert not self.training
        features = self.bone(images)
        assert (features.size(2) % self.num_stripes == 0)
        stripe_h_4 = int(features.size(2) / 4)
        local_4_feat_list = []
        final_feat_list = []
        logits_list = []
        rest_4_feat_list = []
        logits_local_rest_list = []
        logits_local_list = []
        logits_rest_list = []
        logits_global_list = []
        for i in range(4):
            local_4_feat = F.max_pool2d(
                features[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],
                (stripe_h_4, features.size(-1)))
            local_4_feat_list.append(local_4_feat)
        global_4_max_feat = F.max_pool2d(features,
                                         (features.size(2), features.size(3)))
        global_4_rest_feat = (local_4_feat_list[0] + local_4_feat_list[1] + local_4_feat_list[2]
                              + local_4_feat_list[3] - global_4_max_feat) / 3
        global_4_max_feat = self.global_4_max_conv_list[0](global_4_max_feat)
        global_4_rest_feat = self.global_4_rest_conv_list[0](global_4_rest_feat)
        global_4_max_rest_feat = self.global_4_pooling_conv_list[0](
            torch.cat((global_4_max_feat, global_4_rest_feat), 1))
        global_4_feat = (global_4_max_feat + global_4_max_rest_feat).squeeze(3).squeeze(2)
        for i in range(4):
            rest_4_feat_list.append((local_4_feat_list[(i + 1) % 4]
                                     + local_4_feat_list[(i + 2) % 4]
                                     + local_4_feat_list[(i + 3) % 4]) / 3)

        for i in range(4):
            local_4_feat = self.local_4_conv_list[i](local_4_feat_list[i]).squeeze(3).squeeze(2)
            input_rest_4_feat = self.rest_4_conv_list[i](rest_4_feat_list[i]).squeeze(3).squeeze(2)
            input_local_rest_4_feat = torch.cat((local_4_feat, input_rest_4_feat), 1).unsqueeze(2).unsqueeze(3)
            local_rest_4_feat = self.relation_4_conv_list[i](input_local_rest_4_feat)
            local_rest_4_feat = (local_rest_4_feat
                                 + local_4_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            final_feat_list.append(local_rest_4_feat)
            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_4_list[i](local_rest_4_feat))
                logits_local_list.append(self.fc_local_4_list[i](local_4_feat))
                logits_rest_list.append(self.fc_rest_4_list[i](input_rest_4_feat))
        final_feat_list.append(global_4_feat)
        final_feat_list1 = torch.tensor([item.cpu().detach().numpy() for item in final_feat_list]).cuda()
        final_feat = final_feat_list1.permute(1, 2, 0)
        global_feat = self.heads(final_feat)

        stn = self.stn(images)
        head = stn[0][0]
        grid_list = stn[1][0]
        head_feat = self.hsa(head)
        head_pool_feat = self.heatpool(head_feat)

        hf_1 = self.HAN_1(head_feat[:, :, 0:4, :])
        hf1_pool_feat = self.hsa_1_pool(hf_1)
        hf_2 = self.HAN_2(head_feat[:, :, 4:8, :])
        hf2_pool_feat = self.hsa_2_pool(hf_2)
        hf_3 = self.HAN_3(head_feat[:, :, 8:12, :])
        hf3_pool_feat = self.hsa_3_pool(hf_3)
        hf_4 = self.HAN_4(head_feat[:, :, 12:16, :])
        hf4_pool_feat = self.hsa_4_pool(hf_4)
        hf = torch.cat((hf1_pool_feat, hf2_pool_feat, hf3_pool_feat, hf4_pool_feat), dim=1)
        hm1 = torch.sub(input=hf1_pool_feat, alpha=1, other=head_pool_feat)
        hm2 = torch.sub(input=hf2_pool_feat, alpha=1, other=head_pool_feat)
        hm3 = torch.sub(input=hf3_pool_feat, alpha=1, other=head_pool_feat)
        hm4 = torch.sub(input=hf4_pool_feat, alpha=1, other=head_pool_feat)
        hm = torch.cat((hm1, hm2, hm3, hm4), dim=1)


        with torch.no_grad():
            score_maps, keypoints_confidence, _ = self.scoremap_computer(images)
        feature_vector_list, keypoints_confidence = compute_local_features(
            self._cfg, features, score_maps, keypoints_confidence)
        bned_feature_vector_list, cls_score_list = self.bnclassifiers(feature_vector_list)

        gcned_feature_vector_list = self.gcn(feature_vector_list)
        bned_gcned_feature_vector_list, gcned_cls_score_list = self.bnclassifiers2(gcned_feature_vector_list)

        gf = global_feat
        cls_score_list1 = torch.tensor([item.cpu().detach().numpy() for item in cls_score_list]).cuda()
        gcned_cls_score_list1 = torch.tensor([item.cpu().detach().numpy() for item in gcned_cls_score_list]).cuda()
        pred_feat = torch.cat((gf.unsqueeze(2).repeat(1,1,14),hf.unsqueeze(2).repeat(1,1,14), hm.unsqueeze(2).repeat(1,1,14),cls_score_list1.permute(1,2,0), gcned_cls_score_list1.permute(1,2,0)),dim=1)
        pred_feat = self.final_head(pred_feat)

        return nn.functional.normalize(pred_feat)

    def HAN_1(self, x):
        c_att = self.han_c_att_1(self.han_pool_1(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat

    def HAN_2(self, x):
        c_att = self.han_c_att_2(self.han_pool_2(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat

    def HAN_3(self, x):
        c_att = self.han_c_att_3(self.han_pool_3(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat

    def HAN_4(self, x):
        c_att = self.han_c_att_4(self.han_pool_4(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat




