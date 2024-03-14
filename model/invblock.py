import torch
import torch.nn as nn
from model.denseblock import Dense
import config as c


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1*4
            self.split_len2 = in_2*4
        self.clamp = clamp

        # 上面提示了 subnet_constructor的类型是 Dense, Dense 是一个用来提特征的 5层 残差网络
        # Dense 的两个参数名字叫 input 和 output ，显然 表示了 Dense 网络 输入和输出的shape

        # 这里 的4份 subnet_constructor 可以看出 r y 是一组， f,p是一组


        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x):
        # 将传进来的 24 channels dwt 换回 两个12 channels的
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        t2 = self.f(x2)
        s2 = self.p(x2)
        # y1 = x1  + t2
        # y1 = x1

        # y1 = self.e(s2) * x1
        y1 = self.e(s2) * x1 + t2

        s1, t1 = self.r(y1), self.y(y1)
        y2 = self.e(s1) * x2 + t1

        return torch.cat((y1, y2), 1)
