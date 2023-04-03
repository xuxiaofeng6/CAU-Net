# -*- coding: utf-8 -*-
import torch.nn as nn
from blocks.unet_utils import inconv, down_block, up_block
from blocks.utils import get_block, get_norm
from blocks.attention_module import AIM


class B_Net(nn.Module):
    def __init__(self, in_ch=1, base_ch=30, scale=[2,2,2,2], kernel_size=[3,3,3,3], num_classes=1, block='BasicBlock', pool=True, norm='bn'):
        super().__init__()

        num_block = 1
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.aim1 = AIM(num_channels=base_ch)
        self.aim2 = AIM(num_channels=2 * base_ch)
        self.aim3 = AIM(num_channels=4 * base_ch)
        self.aim4 = AIM(num_channels=8 * base_ch)
        self.aim5 = AIM(num_channels=10 * base_ch)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)


    def forward(self, x): 
    
        x1 = self.inc(x)
        x1 = self.aim1(x1)
        x2 = self.down1(x1)
        x2 = self.aim2(x2)
        x3 = self.down2(x2)
        x3 = self.aim3(x3)
        x4 = self.down3(x3)
        x4 = self.aim4(x4)
        x5 = self.down4(x4)
        x5 = self.aim5(x5)

        out = self.up1(x5, x4)
        out = self.aim4(out)
        out = self.up2(out, x3)
        out = self.aim3(out)
        out = self.up3(out, x2)
        out = self.aim2(out)
        out = self.up4(out, x1)
        out = self.aim1(out)
        out = self.outc(out)

        return out
