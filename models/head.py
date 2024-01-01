# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


class Channel_attention(nn.Module):
    def __init__(self, hidden_channel=16):
        super(Channel_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_channel = hidden_channel

    def forward(self, x):
        x_new = torch.zeros([x.size(0), self.hidden_channel, x.size(2), x.size(3)]).cuda()
        for i in range(x.size(0)):
            x_batch = x[i, :, :, :].unsqueeze(0)
            weight = self.avg_pool(x_batch).squeeze()
            weight_list = list(weight)
            import pandas as pd
            weight_index = pd.Series(weight_list).sort_values().index[:int(self.hidden_channel)]
            x_new[i, :, :, :] = x_batch.squeeze(0)[weight_index, :, :]
        return x_new


class DeUp_Cat(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(output_channels * 2, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        y = self.bn(y)
        y = self.relu(y)
        return y


class DeDown_Cat(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeDown_Cat, self).__init__()
        self.conv0 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(output_channels * 2, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        prev = self.conv0(prev)
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        y = self.bn(y)
        y = self.relu(y)
        return y


class frequency_decoder_fancy(nn.Module):
      def __init__(self, input_channel_list=[16, 32, 64, 512], output_channels=4, hidden_channel=16, hidden_channel_low=32, image_size=128):
        super(frequency_decoder_fancy, self).__init__()
        self.hidden_channel = hidden_channel
        self.image_size = image_size
        self.hidden_channel_low = hidden_channel_low


        self.DeUp1 = DeUp_Cat(input_channels=input_channel_list[3], output_channels=input_channel_list[2])
        self.DeUp2 = DeUp_Cat(input_channels=input_channel_list[2], output_channels=input_channel_list[1])
        self.DeUp3 = DeUp_Cat(input_channels=input_channel_list[1], output_channels=input_channel_list[0])

        self.DeDown1 = DeDown_Cat(input_channels=input_channel_list[0], output_channels=input_channel_list[1])
        self.DeDown2 = DeDown_Cat(input_channels=input_channel_list[1], output_channels=input_channel_list[2])
        self.DeDown3 = DeDown_Cat(input_channels=input_channel_list[2], output_channels=input_channel_list[3])


        self.Channel_attention = nn.Conv2d(input_channel_list[3], hidden_channel_low, kernel_size=1)
        self.output_conv_high = nn.Conv2d(self.hidden_channel, output_channels, kernel_size=1)
        self.output_conv_low = nn.Conv2d(hidden_channel_low, output_channels, kernel_size=1)
        self.image_size_low = int(image_size / 4)

        self.learning_weight_multipy_high = nn.Parameter(torch.ones([1, self.hidden_channel, image_size, image_size], requires_grad=True))
        self.learning_weight_plus_high = nn.Parameter(torch.ones([1, self.hidden_channel, image_size, image_size], requires_grad=True))
        self.learning_weight_multipy_low = nn.Parameter(torch.ones([1, self.hidden_channel_low, self.image_size_low, self.image_size_low], requires_grad=True))
        self.learning_weight_plus_low = nn.Parameter(torch.ones([1, self.hidden_channel_low, self.image_size_low, self.image_size_low], requires_grad=True))

      def forward(self, features):
        b = features[0].size(0)

        learning_weight_multiply_high = self.learning_weight_multipy_high.expand(b, self.hidden_channel, self.image_size, self.image_size)
        learning_weight_plus_high = self.learning_weight_plus_high.expand(b, self.hidden_channel, self.image_size, self.image_size)
        learning_weight_multipy_low = self.learning_weight_multipy_low.expand(b, self.hidden_channel_low, self.image_size_low, self.image_size_low)
        learning_weight_plus_low = self.learning_weight_plus_low.expand(b, self.hidden_channel_low, self.image_size_low, self.image_size_low)

        # low-level
        y1_low = self.DeUp1(features[3], features[2])
        y2_low = self.DeUp2(y1_low, features[1])
        y3_low = self.DeUp3(y2_low, features[0])
        x_high = y3_low 

        # high-level
        y1_high = self.DeDown1(features[0], features[1])
        y2_high = self.DeDown2(y1_high, features[2])
        y3_high = self.DeDown3(y2_high, features[3])
        y3_high = self.Channel_attention(y3_high)
        x_low = y3_high

        x_res_high = x_high
        x_high = fft.fft2(x_high, dim=[2,3], norm='ortho')
        x_high = x_high * learning_weight_multiply_high + learning_weight_plus_high
        x_high = fft.ifft2(x_high, dim=[2,3], norm='ortho')
        x_high = x_high.abs() + x_res_high
        x_high = self.output_conv_high(x_high)

        x_low = F.interpolate(x_low, size=(self.image_size_low, self.image_size_low), mode='bilinear', align_corners=True)
        x_res_low = x_low
        x_low = fft.fft2(x_low, dim=[2,3], norm='ortho')
        x_low = x_low * learning_weight_multipy_low + learning_weight_plus_low
        x_low = fft.ifft2(x_low, dim=[2,3], norm='ortho')
        x_low = x_low.abs() + x_res_low
        x_low = self.output_conv_low(x_low)

        return [x_high, x_low]


def mask_foreground_pixel(image, mask_ratio, mask_token):
    B, C, h, w = image.size(0), image.size(1), image.size(2), image.size(3)
    pixel_num = h * w * B
    image = torch.einsum('bchw->cbhw', image)
    image = image.reshape(shape=(C, pixel_num))

    # foreground extract
    foreground_1 = torch.zeros(pixel_num)
    foreground_2 = torch.zeros(pixel_num)
    foreground_3 = torch.zeros(pixel_num)
    foreground_4 = torch.zeros(pixel_num)
    foreground = torch.zeros(pixel_num)

    foreground_1[torch.where(image[0, :] != 0)] = 1
    foreground_2[torch.where(image[1, :] != 0)] = 1
    foreground_3[torch.where(image[2, :] != 0)] = 1
    foreground_4[torch.where(image[3, :] != 0)] = 1
    foreground_sum = foreground_2 + foreground_1 + foreground_3 + foreground_4
    foreground[torch.where(foreground_sum == 4)] = 1

    pixel_mask_idx = torch.nonzero(foreground)
    mask_pixel_count = int(pixel_mask_idx.size(0) * mask_ratio)
    idx = torch.randperm(pixel_mask_idx.size(0))[:mask_pixel_count]
    pixel_mask_idx = pixel_mask_idx[idx]
    pixel_mask = torch.zeros(pixel_num, dtype=int)
    pixel_mask[pixel_mask_idx] = 1
    pixel_mask_tokens = mask_token.expand(C, pixel_num)
    pixel_mask = pixel_mask.type_as(pixel_mask_tokens).unsqueeze(0).expand(C, -1)
    image = image * (1. - pixel_mask) + pixel_mask_tokens * pixel_mask
    image = image.reshape(shape=(C, B, h, w))
    image = torch.einsum('cbhw->bchw', image)
    return image
