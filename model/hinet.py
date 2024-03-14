from model import *
from model.invblock import INV_block_affine
import torch.nn as nn
class Hinet_stage(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self):
        super(Hinet_stage, self).__init__()

        self.inv1 = INV_block_affine()
        self.inv2 = INV_block_affine()


    def forward(self, x):
        # 输入一张图 x ,24 通道
        out = self.inv1(x)

        out = self.inv2(out)

        return out
