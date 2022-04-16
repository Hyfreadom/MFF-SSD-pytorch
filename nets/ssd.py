import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nets.vgg import vgg as add_vgg


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == 'vgg':
        # Block 6
        # 19,19,1024 -> 19,19,256 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        # 10,10,512 -> 10,10,128 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 5,5,128 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        
        # Block 9
        # 3,3,256 -> 3,3,128 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return nn.ModuleList(layers)

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels,kernel_size,stride,padding):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, int(in_channels/2), kernel_size, stride,padding)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels,kernel_size=3, stride=1,padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x2 = self.up(x2)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1


class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained = False):
        super(SSD300, self).__init__()
        self.num_classes    = num_classes
        if backbone_name    == "vgg":
            self.vgg        = add_vgg(pretrained)
            self.extras     = add_extras(1024, backbone_name)
            self.decoder4   = Decoder(in_channels=256,middle_channels=256+128,out_channels=256,
                                    kernel_size=3,stride=1,padding=0)
            self.decoder3   = Decoder(256,256+128,256,3,1,0)
            self.decoder2   = Decoder(256,512+128,512,2,2,0)
            self.decoder1   = Decoder(512,1024+256,1024,3,2,1)
            self.decoder0   = Decoder(1024,512+512,512,2,2,0)
            self.L2Norm     = L2Norm(512,20)
            mbox            = [4, 6, 6, 6, 4, 4]
            
            loc_layers      = []
            conf_layers     = []

            loc_layers      += [nn.Conv2d(128, 4 * 4, kernel_size = 3, padding = 1),
                                nn.Conv2d(256, 4 * 4, kernel_size = 3, padding = 1),
                                nn.Conv2d(512, 4 * 4, kernel_size = 3, padding = 1),
                                nn.Conv2d(1024, 4 * 4, kernel_size = 3, padding = 1)]
            conf_layers     += [nn.Conv2d(128, 4 * num_classes, kernel_size = 3, padding = 1),
                                nn.Conv2d(256, 4 * num_classes, kernel_size = 3, padding = 1),
                                nn.Conv2d(512, 4 * num_classes, kernel_size = 3, padding = 1),
                                nn.Conv2d(1024, 4 * num_classes, kernel_size = 3, padding = 1)]
        self.loc            = nn.ModuleList(loc_layers)
        self.conf           = nn.ModuleList(conf_layers)
        self.backbone_name  = backbone_name
        
    def forward(self, x):
        #---------------------------#
        #   x是300,300,3
        #---------------------------#
        sources = list()
        uleft   = list()
        loc     = list()
        conf    = list()


        #---------------------------#
        #   先经过vgg网络部分
        #---------------------------#


        #   获得U型网络第1层的内容   
        #   shape为38,38,512
        if self.backbone_name == "vgg":
            for k in range(0,23):
                x = self.vgg[k](x)
        u1 = self.L2Norm(x)
        uleft.append(u1)

        #   获得U型网络第2层的内容   
        #   shape为19,19,1024
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)      #<class 'torch.Tensor'>  torch.Size([1, 1024, 19, 19])
        u2 = x
        uleft.append(u2)

        #-------------------------------------------------------------#
        #   经过 extra 网络
        #   第1层、第3层、第5层、第7层的特征层  可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#              
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    uleft.append(x)
        u_right_5   = uleft[5]
        u_right_4   = self.decoder4(uleft[4],u_right_5)
        u_right_3   = self.decoder3(uleft[3],u_right_4)
        u_right_2   = self.decoder2(uleft[2],u_right_3)
        u_right_1   = self.decoder1(uleft[1],u_right_2)
        u_right_0   = self.decoder0(uleft[0],u_right_1)
        sources = [u_right_0,u_right_1,u_right_2,u_right_3,u_right_4,u_right_5]
        for iter in sources:
            print(iter.shape)


        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        #-------------------------------------------------------------# 
        print(len(sources))
        print(len(self.loc))
        print(len(self.conf))    
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #-------------------------------------------------------------#
        #   进行reshape方便堆叠
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        #-------------------------------------------------------------#     
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output

if __name__=="__main__":
    print("ok")