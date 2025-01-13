import torch
from modules import upsample, MFEM, norm_layer, Fusion
import torch.nn.functional as F
from HolisticAttention import HA
from torch import nn
from torchvision import models
from pvtv2 import pvt_v2_b2
from options import opt

CHANNELS = [64, 128, 320, 512]


def upsample_sigmoid(pred_list, target_size):
    if target_size:
        for i in range(len(pred_list)):
            pred_list[i] = torch.sigmoid(upsample(pred_list[i], target_size))
    else:
        for i in range(len(pred_list)):
            pred_list[i] = torch.sigmoid(pred_list[i])
    return pred_list


class PositioningDecoder(nn.Module):
    def __init__(self, config):
        super(PositioningDecoder, self).__init__()
        self.compress4 = nn.Sequential(
            nn.Conv2d(CHANNELS[3], 64, 1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )
        self.compress3 = nn.Sequential(
            nn.Conv2d(CHANNELS[2], 64, 1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )
        self.compress2 = nn.Sequential(
            nn.Conv2d(CHANNELS[1], 64, 1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        self.locate2 = MFEM(64, 64)
        self.locate3 = MFEM(64, 64)
        self.locate4 = MFEM(64, 64, nl=True)

        self.predict = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x2, x3, x4):
        x2 = self.compress2(x2)
        x3 = self.compress3(x3)
        x4 = self.compress4(x4)

        x4 = self.locate4(x4)
        x3 = x3 + upsample(x4, x3.shape[2:])
        x3 = self.locate3(x3)
        x2 = x2 + upsample(x3, x2.shape[2:])
        x2 = self.locate2(x2)

        attention_map = torch.sigmoid(self.predict(x2))
        edge = torch.abs(F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1) - attention_map)
        return attention_map, edge


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.encoder = pvt_v2_b2()

        self.decoder1 = PositioningDecoder(config)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.ha = HAM()
        from modules import BEMAtt as FusionBlock

        self.fuse1 = FusionBlock()
        self.fuse2 = FusionBlock()
        self.fuse3 = FusionBlock()
        self.fuse4 = Fusion()

        self.compress3 = MFEM(CHANNELS[3], 64)
        self.compress2 = MFEM(CHANNELS[2], 64)
        self.compress1 = MFEM(CHANNELS[1], 64)
        self.compress0 = MFEM(CHANNELS[0], 64)

        # self._initialize_weight()

    def crop_features(self, input_size, tmp_size, prediction, boundary, features, ratio):
        """
        According to the prediction maps, new feature maps can be cropped from the input features
        :param input_size: the size of the input image
        :param tmp_size: the size of the x1 feature
        :param prediction: prediction maps generated from the first branch
        :param boundary: boundary prediction maps generated from the first branch
        :param features: x1
        :param ratio: since the first-branch prediction maps may not cover the entire the target,
                    enlarge the bounding box is helpful to guarantee better performace, note ratio > 1
        :return: cropped features, the coordinate of the cropped bounding box, cropped mask and cropped edge
        """
        N, C, H, W = features.shape
        cropped_features, cropped_masks, cropped_edges, cropped_coors = [], [], [], []
        _prediction = upsample(prediction, (tmp_size, tmp_size))
        _boundary = upsample(boundary, (tmp_size, tmp_size))
        for i in range(N):
            i_prediction = _prediction[i]
            # 将预测结果锚框投射到resnet特征x1上时，输入坐标
            i_low, i_high, i_left, i_right = self.fetchBBox(tmp_size, i_prediction, ratio)
            cropped_coor = [i_low, i_high, i_left, i_right]
            cropped_coors.append(cropped_coor)

            cropped_feature = features[i, :, i_low: i_high + 1, i_left: i_right + 1]
            # print(i_low, i_high, i_left, i_right, cropped_feature.shape)
            cropped_feature = upsample(cropped_feature.unsqueeze(0), (opt.size2, opt.size2))
            cropped_features.append(cropped_feature)

            cropped_mask = _prediction[i, :, i_low: i_high + 1, i_left: i_right + 1]
            cropped_mask = upsample(cropped_mask.unsqueeze(0), cropped_feature.shape[2:])
            cropped_masks.append(cropped_mask)

            cropped_edge = _boundary[i, :, i_low: i_high + 1, i_left: i_right + 1]
            cropped_edge = upsample(cropped_edge.unsqueeze(0), cropped_feature.shape[2:])
            cropped_edges.append(cropped_edge)

        cropped_features = torch.cat(cropped_features, dim=0)
        cropped_masks = torch.cat(cropped_masks, dim=0)
        cropped_edges = torch.cat(cropped_edges, dim=0)
        return cropped_features, cropped_coors, cropped_masks, cropped_edges

    def fetchBBox(self, input_size, prediction, ratio):
        """
        Given a prediction map, output the bounding box coordinate
        :param input_size: the height or width of resnet feature x1
        :param prediction:
        :param ratio:
        :return:
        """
        prediction = prediction.squeeze()
        prediction = (prediction - prediction.min()) / \
                     (prediction.max() - prediction.min() + 1e-10)
        prediction = (prediction > 0.5).float()
        prediction_w = prediction.sum(dim=0)
        prediction_w_ = prediction_w > 0
        prediction_w_coor = torch.nonzero(prediction_w_ == True).squeeze()
        left, right = prediction_w_coor.min(), prediction_w_coor.max()

        prediction_h = prediction.sum(dim=1)
        prediction_h_ = prediction_h > 0
        prediction_h_coor = torch.nonzero(prediction_h_ == True).squeeze()
        low, high = prediction_h_coor.min(), prediction_h_coor.max()

        core_w, core_h = (left + right) // 2, (high + low) // 2
        length = (max(high - low + 1, right - left + 1) * ratio).int()
        new_left, new_right = core_w - length // 2, core_w + length // 2
        new_low, new_high = core_h - length // 2, core_h + length // 2

        # l, r, l, h should > 0 and < image bound
        new_left = min(max(0, new_left), input_size - 3)
        new_right = min(max(new_left + 2, new_right), input_size - 1)
        new_low = min(max(0, new_low), input_size - 3)
        new_high = min(max(new_low + 2, new_high), input_size - 1)

        if new_left == new_right or new_low == new_high:
            print("left", new_left, "low", new_low)

        return new_low, new_high, new_left, new_right

    def restore(self, input_size, tmp_size, cropped_pred, bbox_):
        N, C, H, W = cropped_pred.shape
        # 由于在第二个分支中，我们是从x1特征图中，切出来一个bbox_大小的图，然后进行放大，再进行预测
        # 因此，再获得第二个分支的预测结果之后，我们需要先将预测结果缩小到bbox，并且按照之前的bbox坐标填充到一个
        # x1大小的全0预测结果中，最后再将x1大小的预测结果放大到原来的大小。
        output_tensor = torch.zeros(size=(N, 1, tmp_size, tmp_size)).to(cropped_pred.device)
        for i in range(N):
            i_cropped_pred = cropped_pred[i]
            i_low, i_high, i_left, i_right = bbox_[i]
            i_cropped_pred_resize = upsample(i_cropped_pred.unsqueeze(0),
                                             (i_high - i_low + 1, i_right - i_left + 1))
            output_tensor[i, :, i_low:i_high + 1, i_left:i_right + 1] = i_cropped_pred_resize
        output_tensor = upsample(output_tensor, (input_size, input_size))
        return output_tensor

    def forward(self, x):
        B = x.shape[0]
        # encoder stage1
        ## RGB Branch
        x1, x_H, x_W = self.encoder.patch_embed1(x)
        for i, blk in enumerate(self.encoder.block1):
            x1 = blk(x1, x_H, x_W)
        x1 = self.encoder.norm1(x1)
        x1 = x1.reshape(B, x_H, x_W, -1).permute(0, 3, 1, 2).contiguous()

        x1_1 = self.pool(x1)

        # encoder stage2
        ## RGB Branch
        x2_1, x_H, x_W = self.encoder.patch_embed2_1(x1_1)
        for i, blk in enumerate(self.encoder.block2_1):
            x2_1 = blk(x2_1, x_H, x_W)
        x2_1 = self.encoder.norm2_1(x2_1)
        x2_1 = x2_1.reshape(B, x_H, x_W, -1).permute(0, 3, 1, 2).contiguous()
        # layer2 merge end

        # encoder stage3
        ## RGB Branch
        x3_1, x_H, x_W = self.encoder.patch_embed3_1(x2_1)
        for i, blk in enumerate(self.encoder.block3_1):
            x3_1 = blk(x3_1, x_H, x_W)
        x3_1 = self.encoder.norm3_1(x3_1)
        x3_1 = x3_1.reshape(B, x_H, x_W, -1).permute(0, 3, 1, 2).contiguous()

        # encoder stage4
        ## RGB Branch
        x4_1, x_H, x_W = self.encoder.patch_embed4_1(x3_1)
        for i, blk in enumerate(self.encoder.block4_1):
            x4_1 = blk(x4_1, x_H, x_W)
        x4_1 = self.encoder.norm4_1(x4_1)
        x4_1 = x4_1.reshape(B, x_H, x_W, -1).permute(0, 3, 1, 2).contiguous()

        attention_map, edge = self.decoder1(x2_1, x3_1, x4_1)

        cropped_features, bbox, cropped_mask, cropped_edge = self.crop_features(
            input_size=opt.trainsize,
            tmp_size=opt.trainsize // 4,
            prediction=attention_map,
            boundary=edge,
            features=x1,
            ratio=opt.expand_ratio
        )

        ###### Second Branch #######
        # stage2
        x2_2, x_H, x_W = self.encoder.patch_embed2_2(cropped_features)
        for i, blk in enumerate(self.encoder.block2_2):
            x2_2 = blk(x2_2, x_H, x_W)
        x2_2 = self.encoder.norm2_2(x2_2)
        x2_2 = x2_2.reshape(B, x_H, x_W, -1).permute(0, 3, 1, 2).contiguous()
        x2_2 = x2_2

        # stage 3
        x3_2, H, W = self.encoder.patch_embed3_2(x2_2)
        for i, blk in enumerate(self.encoder.block3_2):
            x3_2 = blk(x3_2, H, W)
        x3_2 = self.encoder.norm3_2(x3_2)
        x3_2 = x3_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x4_2, H, W = self.encoder.patch_embed4_2(x3_2)
        for i, blk in enumerate(self.encoder.block4_2):
            x4_2 = blk(x4_2, H, W)
        x4_2 = self.encoder.norm4_2(x4_2)
        x4_2 = x4_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x1_1_compress = self.compress0(x1)
        x2_2_compress = self.compress1(x2_2)
        x3_2_compress = self.compress2(x3_2)
        x4_2_compress = self.compress3(x4_2)

        x4_refined, mask4, edge4 = self.fuse1(x4_2_compress, None, cropped_mask, cropped_edge)
        x3_refined, mask3, edge3 = self.fuse2(x3_2_compress, x4_refined, mask4, edge4)
        x2_refined, mask2, edge2 = self.fuse3(x2_2_compress, x3_refined, mask3, edge3)

        mask_list = [mask2, mask3, mask4]
        edge_list = [edge2, edge3, edge4]

        for i in range(len(mask_list)):
            mask_list[i] = self.restore(input_size=opt.trainsize, tmp_size=opt.trainsize // 4,
                                        cropped_pred=mask_list[i], bbox_=bbox)
            edge_list[i] = self.restore(input_size=opt.trainsize, tmp_size=opt.trainsize // 4,
                                        cropped_pred=edge_list[i], bbox_=bbox)
        mask1, edge1 = self.fuse4(mask_list[0], edge_list[0], x1_1_compress)
        mask1 = upsample(mask1, self.config.trainsize)
        edge1 = upsample(edge1, self.config.trainsize)
        mask_list = [mask1, *mask_list]
        edge_list = [edge1, *edge_list]

        return upsample(attention_map, self.config.trainsize), upsample(edge, self.config.trainsize), \
            mask_list, edge_list, cropped_mask, cropped_edge

    def _initialize_weight(self):
        print("start initialization.......")
        path = './models/pvt_v2_b2.pth'
        pretrained_dict = torch.load(path)
        # pretrained_dict = save_model.state_dict()
        all_params = {}
        for k, v in self.encoder.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v

        assert len(all_params.keys()) == len(self.encoder.state_dict().keys())
        self.encoder.load_state_dict(all_params)


if __name__ == '__main__':
    from options import opt

    model = Net(opt).cuda()
    img = torch.randn(2, 3, 512, 512).cuda()
    # model.load_state_dict(torch.load('models/resnet/TSNet_epoch_best.pth'))
    with torch.no_grad():
        out = model(img)
