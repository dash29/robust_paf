import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class PELU(nn.Module):
    def __init__(self):
        super(PELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1.]))

    def forward(self, x):
        return x * (x > 0).float() + self.alpha * (torch.exp(x) - 1) * (x <= 0).float()

class PBLU(nn.Module):

    def __init__(self):
        super(PBLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([0.]))
    def forward(self, x):
        return (x > 0).float() * (self.alpha * (torch.sqrt(x ** 2 + 1) - 1) + x)

class PReLU(nn.Module):

    def __init__(self):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        return (x > 0).float() * x + (x < 0).float() * self.alpha * x

class PPReLU(nn.Module):
    def __init__(self):
        super(PPReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1.]))

    def forward(self, x):
        return (x > 0).float() * x * self.alpha

class PSiLU(nn.Module):
    def __init__(self):
        super(PSiLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1.]))

    def forward(self, x):
        return x * F.sigmoid(torch.abs(self.alpha) * x)

class Softplus(nn.Module):
    # we implement softplus and psoftplus using logsigmoid to increase numerical stability
    def __init__(self):
       super(Softplus, self).__init__()
    def forward(self, x):
       return -F.logsigmoid(-x)

class PSoftplus(nn.Module):
    def __init__(self):
        super(PSoftplus, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1.]))

    def forward(self, x):
        #return (torch.abs(self.alpha) * x >= self.threshold).float() * x + (torch.abs(self.alpha) * x < self.threshold).float() * (1/torch.abs(self.alpha) * torch.log(1 + torch.exp(torch.abs(self.alpha) * x)))
        return - (1/self.alpha) * F.logsigmoid(-self.alpha * x)

class PSSiLU(nn.Module):
    def __init__(self, beta=0.1):
        super(PSSiLU, self).__init__()
        self.beta = beta
        self.alpha = nn.Parameter(torch.tensor([1.0]))
    def forward(self, x):
        return x * (F.sigmoid(torch.abs(self.alpha) * x) - self.beta) / (1 - self.beta)

class PSSiLU2(nn.Module):
    def __init__(self):
        super(PSSiLU2, self).__init__()
        self.beta = nn.Parameter(torch.tensor([1e-8]))
        self.alpha = nn.Parameter(torch.tensor([1.0]))
    def forward(self, x):
        return x * (F.sigmoid(torch.abs(self.alpha) * x) - torch.abs(self.beta)) / (1 - torch.abs(self.beta))

def relu(**kwargs):
   return nn.ReLU()
def elu(**kwargs):
   return nn.ELU()
def softplus(**kwargs):
   return Softplus()
def silu(**kwargs):
   return SiLU()
def prelu(**kwargs):
   return PReLU()
def pelu(**kwargs):
   return PELU()
def psilu(**kwargs):
   return PSiLU()
def pblu(**kwargs):
   return PBLU()
def psoftplus(**kwargs):
   return PSoftplus()
def pssilu(beta=0.1):
   return PSSiLU(beta)
def pssilu2(**kwargs):
   return PSSiLU2()
def pprelu(**kwargs):
   return PPReLU()
