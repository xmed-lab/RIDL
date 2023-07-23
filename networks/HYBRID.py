
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                # avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # print(x.shape)
                avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                # max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # print(x.shape)
                # exit()
                max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                assert False, "not implemented"
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                assert False, "not implemented"
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale, scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            # x_out = self.SpatialGate(x_out)
            x_out, scale = self.SpatialGate(x_out)
        return x_out, scale


# import parameter as para


class HYBRID(nn.Module):
    def __init__(self, training, out_channel = 1, in_channel = 1, dropout = 0.3, fconnect = False, fc_layer_num = 2, glb_ft_num = 2):
        super().__init__()

        self.training = training
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.dropout = dropout
        self.fconnect = fconnect
        self.fc_layer_num = fc_layer_num


        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(self.in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        if self.fconnect:
            self.decoder_stage2 = nn.Sequential(
                nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
                # nn.Conv3d(128, 128, 3, 1, padding=1),
                nn.PReLU(128),

                nn.Conv3d(128, 128, 3, 1, padding=1),
                nn.PReLU(128),

                nn.Conv3d(128, 128, 3, 1, padding=1),
                nn.PReLU(128),
            )
        else:
            self.decoder_stage2 = nn.Sequential(
                # nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
                nn.Conv3d(128, 128, 3, 1, padding=1),
                nn.PReLU(128),

                nn.Conv3d(128, 128, 3, 1, padding=1),
                nn.PReLU(128),

                nn.Conv3d(128, 128, 3, 1, padding=1),
                nn.PReLU(128),
            )

        if self.fconnect:
            self.decoder_stage3 = nn.Sequential(
                nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
                # nn.Conv3d(64, 64, 3, 1, padding=1),
                nn.PReLU(64),

                nn.Conv3d(64, 64, 3, 1, padding=1),
                nn.PReLU(64),

                nn.Conv3d(64, 64, 3, 1, padding=1),
                nn.PReLU(64),
            )
        else:
            self.decoder_stage3 = nn.Sequential(
                # nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
                nn.Conv3d(64, 64, 3, 1, padding=1),
                nn.PReLU(64),

                nn.Conv3d(64, 64, 3, 1, padding=1),
                nn.PReLU(64),

                nn.Conv3d(64, 64, 3, 1, padding=1),
                nn.PReLU(64),
            )

        if self.fconnect:
            self.decoder_stage4 = nn.Sequential(
                nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
                # nn.Conv3d(32, 32, 3, 1, padding=1),
                nn.PReLU(32),

                nn.Conv3d(32, 32, 3, 1, padding=1),
                nn.PReLU(32),
            )
        else:
            self.decoder_stage4 = nn.Sequential(
                # nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
                nn.Conv3d(32, 32, 3, 1, padding=1),
                nn.PReLU(32),

                nn.Conv3d(32, 32, 3, 1, padding=1),
                nn.PReLU(32),
            )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, self.out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )


        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, self.out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )


        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, self.out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, self.out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

        self.fc_0 = nn.Linear(256 + glb_ft_num, 256 + glb_ft_num)
        self.fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(256 + glb_ft_num, 1)

        # self.ctr_0 = nn.Linear(256, 256)

        self.cbam_module = CBAM(256)


    def forward(self, inputs, glb_shp_ft):
        if self.in_channel == 1:
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)


        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dropout, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dropout, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dropout, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dropout, self.training)
        
        outputs_encoded = outputs

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)
        
        if self.fconnect:
            outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        else:
            outputs = self.decoder_stage2(torch.cat([short_range6], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        if self.fconnect:
            outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        else:
            outputs = self.decoder_stage3(torch.cat([short_range7], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        if self.fconnect:
            outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        else:
            outputs = self.decoder_stage4(torch.cat([short_range8], dim=1)) + short_range8

        output4 = self.map4(outputs)
        # print("inputs", inputs.shape, "output4", output4.shape, "outputs_encoded", outputs_encoded.shape)

        # outputs_encoded = self.cbam_module(outputs_encoded)
        outputs_encoded, spat_scale = self.cbam_module(outputs_encoded)

        outputs_encoded_avg = outputs_encoded.mean(dim = (-1,-2,-3))

        outputs_encoded_avg_wglb = torch.cat((outputs_encoded_avg, glb_shp_ft), dim = -1)
        # print(outputs_encoded_avg_wglb.shape)
        # exit()
        if self.fc_layer_num == 2:
            output_cls_0 = self.fc_0(outputs_encoded_avg_wglb)
            output_cls_1 = self.fc_1(output_cls_0)
            output_cls = self.fc_2(output_cls_1)
        elif self.fc_layer_num == 1:
            output_cls = self.fc_2(outputs_encoded_avg_wglb)
        else:
            assert False, "wrong fc_layer_num"


        if self.training is True:
            return output1, output2, output3, output4, output_cls, outputs_encoded, spat_scale
        else:
            return output4, outputs_encoded, output_cls, spat_scale#, output_ctr_1


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        try:
            nn.init.constant_(module.bias.data, 0)
        except:
            print(module.bias)
