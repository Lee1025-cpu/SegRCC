import torch.nn as nn
from .utils_for_unet_grid_attention_3D import UnetConv3, UnetUp3, UnetGridGatingSignal3, UnetConv3_t, UnetUp3_t
import torch.nn.functional as F
from Now_state.segmentation.models.grid_attention_layer import GridAttentionBlock3D
from found.structure.Attention_Gated_Networks_master.models.networks_other import init_weights


class AttentionUnet(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=1,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, dropout=0.3):
        super(AttentionUnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [1, 2, 4, 8, 16]
        filters = [int(x * self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.gating = UnetGridGatingSignal3(filters[4], filters[3], kernel_size=(1, 1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.dropout = nn.Dropout3d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)      # 1025MB 2409MB 2245MB
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # g_conv4, att4 = self.attentionblock4(conv4, gating)
        # g_conv3, att3 = self.attentionblock3(conv3, gating)
        # g_conv2, att2 = self.attentionblock2(conv2, gating)
        g_conv4 = self.attentionblock4(conv4, gating)   # 1765MB 2041
        g_conv3 = self.attentionblock3(conv3, gating)
        g_conv2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)      # 1889MB  2237
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)     # 2699MB 3441
        final = self.dropout(final)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class AttentionUnet_t(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=1,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, dropout=0.3):
        super(AttentionUnet_t, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [1, 2, 4, 8, 16]
        filters = [int(x * self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3_t(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = UnetConv3_t(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = UnetConv3_t(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = UnetConv3_t(filters[2], filters[3], self.is_batchnorm)
        self.center = UnetConv3_t(filters[3], filters[4], self.is_batchnorm)

        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.gating = UnetGridGatingSignal3(filters[4], filters[3], kernel_size=(1, 1, 1),
                                            is_batchnorm=self.is_batchnorm)

        self.conv5 = UnetConv3_t(filters[2], filters[1], self.is_batchnorm)
        self.conv6 = UnetConv3_t(filters[1], filters[0], self.is_batchnorm)

        # attention blocks
        self.attentionblock1 = GridAttentionBlock3D(in_channels=filters[0], gating_channels=filters[1],
                                                    inter_channels=filters[0], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock2 = GridAttentionBlock3D(in_channels=filters[1], gating_channels=filters[2],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample,
                                                    mode=nonlocal_mode)

        # upsampling
        self.up_concat4 = UnetUp3_t(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UnetUp3_t(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UnetUp3_t(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UnetUp3_t(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.dropout = nn.Dropout3d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)      #
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Upscaling Part (Decoder) and Attention Mechanism
        g_conv4 = self.attentionblock4(conv4, gating)   #
        _, up4 = self.up_concat4(g_conv4, center)

        g_conv3 = self.attentionblock3(conv3, up4)
        mid_3, up3 = self.up_concat3(g_conv3, up4)

        g_conv2 = self.attentionblock2(conv2, up3)
        mid_2, up2 = self.up_concat2(g_conv2, up3)

        g_conv1 = self.attentionblock1(conv1, up2)
        _, up1 = self.up_concat1(g_conv1, up2)

        mid_3_0 = self.conv5(self.up_sample(mid_3))
        mid_2_0 = self.conv6(self.up_sample(mid_2 + mid_3_0))
        final = self.final(up1 + mid_2_0)     #

        final = self.dropout(final)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p












