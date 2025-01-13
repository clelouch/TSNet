import torch
from torch import nn
import torch.nn.functional as F
from HolisticAttention import HA
from options import opt


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=False)


def norm_layer(channel, norm_name=opt.norm, _3d=False):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel) if not _3d else nn.BatchNorm3d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class ChannelCompress(nn.Module):
    def __init__(self, in_c, out_c):
        super(ChannelCompress, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=opt.sakernel):
        super(SpatialAttention, self).__init__()
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class NonLocal(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        # channel compression
        self.compress = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            norm_layer(out_channel),
            nn.ReLU(inplace=False)
        )

        # non-local
        temp_c = out_channel // 4
        self.query_conv = nn.Conv2d(in_channels=out_channel, out_channels=temp_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=out_channel, out_channels=temp_c, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # residual connection
        self.conv_res = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            norm_layer(out_channel),
        )

    def forward(self, x):
        # non-local
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        out = F.relu(x + self.conv_res(out1), inplace=False)
        return out


def getBasicBranch(in_channel, tmp_channel, out_channel, pool_kernel):
    conv1 = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, bias=False),
        norm_layer(out_channel),
        nn.ReLU(inplace=False)
    )
    pool1 = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
    conv3 = nn.Sequential(
        nn.Conv2d(tmp_channel, out_channel, 3, 1, 1, bias=False),
        norm_layer(out_channel),
        nn.ReLU(inplace=False)
    )
    dilated = nn.Sequential(
        nn.Conv2d(out_channel, out_channel, 3, 1, 2, dilation=2, bias=False),
        norm_layer(out_channel),
        # nn.ReLU(inplace=False)
    )
    return conv1, pool1, conv3, dilated


# Multi-scale Feature Enhancement Module
class MFEM(nn.Module):
    def __init__(self, in_channel, out_channel, nl=False):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.nl = nl
        if self.nl:
            self.non_local_compress = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=False),
                norm_layer(out_channel),
                nn.ReLU(inplace=False)
            )
            self.non_local = NonLocal(in_channel, out_channel)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            norm_layer(out_channel),
            nn.ReLU(inplace=False)
        )
        self.conv1_branch1, self.pool1_branch1, self.conv3_branch1, self.dilated_branch1 = \
            getBasicBranch(in_channel, out_channel * 2, out_channel, 1)
        self.conv1_branch2, self.pool1_branch2, self.conv3_branch2, self.dilated_branch2 = \
            getBasicBranch(in_channel, out_channel * 2, out_channel, 2)
        self.conv1_branch3, self.pool1_branch3, self.conv3_branch3, self.dilated_branch3 = \
            getBasicBranch(in_channel, out_channel * 2, out_channel, 4)
        self.conv1_branch4, self.pool1_branch4, self.conv3_branch4, self.dilated_branch4 = \
            getBasicBranch(in_channel, out_channel, out_channel, 8)

    def forward(self, x):
        # nonlocal part
        if self.nl:
            x_nonlocal = self.non_local(self.non_local_compress(x))

        # multi-scale part
        # begin
        x_shortcut = self.shortcut(x)
        x1 = self.conv1_branch1(x)
        x2 = self.conv1_branch2(x)
        x3 = self.conv1_branch3(x)
        x4 = self.conv1_branch4(x)

        # merge
        x4 = self.dilated_branch4(self.conv3_branch4(self.pool1_branch4(x4)))
        x3 = self.dilated_branch3(self.conv3_branch3(self.pool1_branch3(
            torch.cat([x3, upsample(x4, x3.shape[2:])], dim=1)
        )))
        x2 = self.dilated_branch2(self.conv3_branch2(self.pool1_branch2(
            torch.cat([x2, upsample(x3, x2.shape[2:])], dim=1)
        )))
        x1 = self.dilated_branch1(self.conv3_branch1(
            torch.cat([x1, upsample(x2, x1.shape[2:])], dim=1)
        ))

        if self.nl:
            out = F.relu(x_shortcut + x1 + x_nonlocal, inplace=False)
        else:
            out = x_shortcut + x1
        return out


# Boundary Enhancement Module using Attention
class BEMAtt(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.fore_boundary1 = nn.Sequential(
            nn.Conv2d(channel + 1, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=False)
        )
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention(7)

        self.pred_boundary = nn.Conv2d(channel, 1, 3, 1, 1)
        self.pred_mask = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel // 2, 3, 1, 1, bias=False),
            norm_layer(channel // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // 2, 1, 3, 1, 1)
        )

    def forward(self, low, high, mask_pred, edge_pred):
        """

        :param f: feature maps
        :param p: prediction map generated from the former stage
        :param b: boundary prediction map generated from the former stage
        :return: refined feature and new predciton maps
        """
        if mask_pred.shape[-1] != low.shape[-1]:
            edge_pred = upsample(edge_pred, low.shape[2:])
        if high is not None:
            low += upsample(high, low.shape[2:])
        feature = low

        # merge feature maps with boudnary map
        feature_b = self.fore_boundary1(torch.cat([feature, edge_pred], dim=1))
        feature_ca = self.ca(feature_b) * feature_b
        feature_sa = self.sa(feature_ca) * feature_ca

        mask_pred = torch.sigmoid(self.pred_mask(feature_sa))
        boundary_pred = torch.sigmoid(self.pred_boundary(feature_sa))
        return feature_sa, mask_pred, boundary_pred


class SplitAndFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.sub_channel = channel // 4
        self.branch1 = SplitAndFusionSubBranch(self.sub_channel)
        self.branch2 = SplitAndFusionSubBranch(self.sub_channel)
        self.branch3 = SplitAndFusionSubBranch(self.sub_channel)
        self.branch4 = SplitAndFusionSubBranch(self.sub_channel)
        self.fusion_cat = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, 1, 1, bias=False),
            norm_layer(self.channel),
            nn.ReLU(inplace=False)
        )
        self.ca = ChannelAttention(channel)

    def forward(self, feature, edge):
        # features = feature.split(4, dim=1)
        feature1, feature2, feature3, feature4 = feature.split(16, dim=1)
        feature1 = self.branch1(feature1, edge)
        feature2 = self.branch2(feature1 + feature2, edge)
        feature3 = self.branch3(feature2 + feature3, edge)
        feature4 = self.branch4(feature3 + feature4, edge)
        feature_cat = self.fusion_cat(torch.cat([feature1, feature2, feature3, feature4], dim=1))
        feature_cat = self.ca(feature_cat) * feature_cat
        return feature_cat


class SplitAndFusionSubBranch(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel + 1, self.channel, 3, 1, 1, bias=False),
            norm_layer(self.channel),
            nn.ReLU(inplace=False)
        )
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention(7)

    def forward(self, feature, edge_pred):
        fused = torch.cat([feature, edge_pred], dim=1)
        fused = self.conv(fused)
        fused_ca = fused * self.ca(fused)
        fused_sa = fused_ca * self.sa(fused_ca)
        return fused_sa


class Fusion(nn.Module):
    def __init__(self, channel=64, kernel=opt.kernel):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channel)
        self.gra1 = SplitAndFusion(channel)
        self.gra2 = nn.Sequential(
            nn.Conv2d(channel + 1, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=False)
        )
        self.pred_edge = nn.Conv2d(channel, 1, 3, 1, 1)

        padding = (kernel - 1) * opt.dilate // 2

        self.feat_boundary = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=False)
        )
        self.fusion_mask_edge = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(kernel, kernel), stride=1, padding=(padding, padding),
                      dilation=(opt.dilate, opt.dilate)),
            norm_layer(channel),
        )
        self.pred_mask = nn.Conv2d(channel, 1, 3, 1, 1)

    def forward(self, mask_pred, edge_pred, x):
        if mask_pred.shape[-1] != x.shape[-1]:
            mask_pred = upsample(mask_pred, x.shape[2:])
            edge_pred = upsample(edge_pred, x.shape[2:])
        x_feat = self.gra1(x, mask_pred)
        x_feat = x_feat * self.ca(x_feat)
        x_feat = x_feat * self.sa(x_feat)

        edge_feat = self.feat_boundary(x_feat)
        edge_feat = self.gra2(torch.cat([edge_feat, edge_pred], dim=1))
        edge_pred = self.pred_edge(edge_feat)

        new_boundary_feature = F.relu(x_feat + self.fusion_mask_edge(x_feat + edge_feat), inplace=False)
        mask_pred = torch.sigmoid(self.pred_mask(new_boundary_feature))
        return mask_pred, torch.sigmoid(edge_pred)


if __name__ == '__main__':
    mm = SplitAndFusion(64)
    a = torch.randn((1, 64, 32, 32))
    b = torch.randn((1, 1, 32, 32))
    out = mm(a, b)
