import torch
from torch import nn
import torch.nn.functional as F

class conv_block(nn.Module):
    # conv -> batch norm -> relu -> conv -> batch norm -> relu
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class unet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters_start=64 ):
        super(unet, self).__init__()
        #self.net = nn.Sequential( 
        #)
        self.conv1 = conv_block(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        
        self.conv2 = conv_block(64, 128)
        self.down2 = nn.MaxPool2d(2)
        
        self.conv3 = conv_block(128, 256)
        self.down3 = nn.MaxPool2d(2)
        
        self.conv4 = conv_block(256, 512)
        self.down4 = nn.MaxPool2d(2)
        
        self.conv5 = conv_block(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # concat conv4 output with up1 and send to conv6
        self.conv6 = conv_block(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = conv_block(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = conv_block(128,64)
        
        self.conv10 = nn.Conv2d(64, n_classes, 1)
        

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
        '''

    def forward(self, input):
        
        x1 = input
        
        z1 = self.conv1(x1)
        
        x2 = self.down1(z1)
        z2 = self.conv2(x2)
        
        x3 = self.down2(z2)
        z3 = self.conv3(x3)
        
        x4 = self.down3(z3)
        z4 = self.conv4(x4)
        
        x5 = self.down4(z4)
        z5 = self.conv5(x5)
        
        
        x6 = torch.cat([z4,self.up1(z5)], dim=1)
        z6 = self.conv6(x6)
        
        x7 = torch.cat([z3, self.up2(z6)], dim=1)
        z7 = self.conv7(x7)
        
        x8 = torch.cat([z2, self.up3(z7)], dim=1)
        z8 = self.conv8(x8)
        
        x9 = torch.cat([z1, self.up4(z8)], dim=1)
        z9 = self.conv9(x9)
        
        z = self.conv10(z9)
        
        z = F.sigmoid(z)
        
        return z
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
