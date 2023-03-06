# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch.utils.model_zoo as model_zoo
import torch
import torch as th
from torch import nn
from torch.autograd import Variable

from sra_modules import SRA_Module
from bam import *
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



# class IBN(nn.Module):
# 	def __init__(self, planes):
# 		super(IBN, self).__init__()
# 		half1 = int(planes / 2)
# 		self.half = half1
# 		half2 = planes - half1
# 		self.IN = nn.InstanceNorm2d(half1, affine=True)  # 实例归一化 对单个样本沿着通道方向进行计算
# 		self.BN = nn.BatchNorm2d(half2)
#
# 	def forward(self, x):
# 		split = torch.split(x, self.half, 1)
# 		out1 = self.IN(split[0].contiguous())
# 		out2 = self.BN(split[1].contiguous())
# 		out = torch.cat((out1, out2), 1)
# 		return out
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)

		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_channels,out_channels, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		# if ibn:
		# 	self.bn1 = IBN(out_channels)
		# else:
		# 	self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels * 4)
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

class BAM_Branch(nn.Module):
	def __init__(self,  last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3],bam_ration=8,bam_bn=True,
		         s_ratio=8, c_ratio=8, d_ratio=8, height=256, width=128, ):
		super(BAM_Branch, self).__init__()
		self.in_channels = 64
		# Networks
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

		# RGA Modules
		self.rga1 = SRA_Module(256, (height//4)*(width//4),
								cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		self.rga2 = SRA_Module(512, (height // 8) * (width // 8),
							  cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		self.rga3 = SRA_Module(1024, (height // 16) * (width // 16),
							  cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		self.rga4 = SRA_Module(2048, (height // 16) * (width // 16),
							  cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		# bam Modules
		self.bam1 = BAM(256, reduction_ratio=bam_ration, bn=bam_bn)
		self.bam2 = BAM(512, reduction_ratio=bam_ration, bn=bam_bn)
		self.bam3 = BAM(1024, reduction_ratio=bam_ration, bn=bam_bn)
		self.bam4 = BAM(2048, reduction_ratio=bam_ration, bn=bam_bn)
		# Load the pre-trained model weights


	def _make_layer(self, block, channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.in_channels != channels * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channels, channels * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(channels * block.expansion),
			)

		layers = []
		layers.append(block(self.in_channels, channels, stride, downsample))
		self.in_channels = channels * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_channels, channels))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.bam1(x)
		x = self.rga1(x)

		x = self.layer2(x)
		x = self.bam2(x)
		#x = self.rga2(x)

		x = self.layer3(x)
		x = self.bam3(x)
		#x = self.rga3(x)

		x = self.layer4(x)
		x = self.bam4(x)
		#x = self.rga4(x)

		return x


	def load_param(self, model_path):
		param_dict = torch.load(model_path)
		for i in param_dict:
			if 'fc' in i:
				continue
			self.state_dict()[i].copy_(param_dict[i])

	def random_init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
# def bam_ibn_a(last_stride, pretrained=False, **kwargs):
#     model = BAM_Branch(last_stride,**kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model
# if __name__ == '__main__':
# 	from IPython import embed
# 	x=torch.randn(3,3,256,128)
# 	moduel=resnet50_rga(num_classes=900, height=256, width=128,loss={'xent'})
# 	o=moduel(x)