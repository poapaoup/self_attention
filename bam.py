import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Avg_pool(nn.Module):
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
class Max_pool(nn.Module):
    def forward(self,x):
        return F.max_pool2d(x,kernel_size=(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
class ChannelGate(nn.Module):
    def __init__(self, gate_channels=20, reduction_ratio=4,bn=True):
        super(ChannelGate, self).__init__()
        self.Avg=Avg_pool()
        self.Max=Max_pool()
        self.gate_channels = gate_channels
        out_planes=gate_channels // reduction_ratio
        self.bn = nn.BatchNorm2d(gate_channels,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, out_planes),
            nn.ReLU(),
            nn.Linear(out_planes, gate_channels)
        )
    def forward(self, x):
        avg=self.Avg(x)
        max=self.Max(x)
        out=avg+max
        out=self.mlp(out)
        out1=out.unsqueeze(2).unsqueeze(3).expand_as(x)
        out1=self.bn(out1)

        return out1
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self, gate_channel=20, reduction_ratio=4,bn=True):
        super(SpatialGate, self).__init__()
        self.conv1=nn.Conv2d(gate_channel,gate_channel//reduction_ratio,kernel_size=1,stride=1)
        self.conv2=nn.Conv2d(2,1,kernel_size=1,stride=1)
        self.ChannelPool=ChannelPool()
        self.bn = nn.BatchNorm2d(gate_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
    def forward(self,x):
        sx=self.conv1(x)
        sx=self.ChannelPool(sx)
        sx=self.conv2(sx)

        sx=sx.expand_as(x)

        sx=self.bn(sx)

        return sx
class BAM(nn.Module):
    def __init__(self, gate_channel, reduction_ratio,bn):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel, reduction_ratio,bn)
        self.spatial_att = SpatialGate(gate_channel,reduction_ratio,bn)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor
if __name__=='__main__':
    from IPython import embed
    x=torch.randn(1,512,32,16)
    o=BAM(gate_channel=20)
    print(o(x))