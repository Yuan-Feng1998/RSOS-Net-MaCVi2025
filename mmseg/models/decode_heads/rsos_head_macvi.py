# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead



class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out

class FeatureFusionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),):
        super().__init__()
        channels = out_channels // scale_factor
        # self.conv0 = ConvModule(
        #     in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv0=depthwise_separable_conv(in_channels,out_channels)
        self.mlp=nn.Sequential(
            ConvModule(
                out_channels,
                channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None),          
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.conv2d = ConvModule(in_channels=2, out_channels=1, kernel_size=3,stride=1,padding=1,norm_cfg=None, act_cfg=None)
    
    def forward(self, feat_fuse0,feat_fuse1,feat_fuse2,feat_fuse3):
        inputs = torch.cat([feat_fuse0,feat_fuse1,feat_fuse2,feat_fuse3], dim=1)
        x = self.conv0(inputs)
        ###ch-attention
        ch_avgout=self.mlp(self.avg_pool(x))
        ch_sigmoid=self.sigmoid(ch_avgout)
        x_attn = x*ch_sigmoid
        #sp-attention
        sp_avgout=torch.mean(x_attn, dim=1, keepdim=True)
        sp_maxout, _ = torch.max(x_attn, dim=1, keepdim=True)
        sp_cat=torch.cat([sp_avgout, sp_maxout], dim=1)
        sp_sigmoid=self.sigmoid(self.conv2d(sp_cat))
        x_attn=x_attn*sp_sigmoid
        return x_attn + x



class FPPM(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1),
            nn.BatchNorm2d(c_),
            nn.ReLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(5*c_, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
        )
        self.m = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        
        x1= self.cv1(x)
        x5=self.global_avg_pool(x1)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3=self.m(y2)
        outputs=torch.cat([x1,y1,y2,y3,x5],dim=1)
        return self.cv2(outputs)

@MODELS.register_module()
class RSOSHead_MaCVi(BaseDecodeHead):
    def __init__(self, fppm=dict(c1=2048, c2=256),ffm=dict(
                     in_channels=1024, out_channels=512, scale_factor=4), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ffm = FeatureFusionModule(**ffm)
        self.fppm=FPPM(**fppm)
        # LFPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels//2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = depthwise_separable_conv(self.channels//2,self.channels//2)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)



    def fppm_forward(self, inputs):

        x = inputs[-1]

        output=self.fppm(x)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.fppm_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        feats=self.ffm(fpn_outs[0],fpn_outs[1],fpn_outs[2],fpn_outs[3])
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
