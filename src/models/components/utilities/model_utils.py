import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import wraps

from einops import rearrange, reduce
import math


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# functions

def exists(val):
    return val is not None


def first(it):
    return it[0]


def default(val, d):
    return val if exists(val) else d



def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


# decorators

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


# tensor functions

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)


# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent=False)
        self.register_buffer('beta', torch.zeros(dim), persistent=False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)


def freeze(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def unfreeze(model):
    for n, p in model.named_parameters():
        p.requires_grad = True


class SoftmaxContrastiveLearning(nn.Module):
    def __init__(
            self,
            *,
            layers=1,
            decoupled_contrastive_learning=False,
            init_temp=10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sims):
        batch = sims.shape[-1]

        if sims.ndim == 2:
            sims = rearrange(sims, 'i j -> 1 i j')

        #dt = 10

        #c = torch.clamp(self.temperatures.exp(), 1, dt)

        sims = sims * self.temperatures.exp()

        cosine_sims_exp = sims.exp()

        numerator = matrix_diag(cosine_sims_exp)

        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))
        #print(torch.min(numerator).item(), torch.min(-log(numerator)).item(), torch.min(log(denominator_i)).item(),
        #      torch.min(log(denominator_j)).item())

        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()


class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sims):
        if sims.ndim == 2:
            sims = rearrange(sims, 'i j -> 1 i j')

        n = sims.shape[-1]
        sims = sims * self.temperatures.exp() + self.bias
        labels = 2 * rearrange(torch.eye(n), 'i j -> 1 i j').to(sims.device) - torch.ones_like(sims)

        return -F.logsigmoid(labels * sims).sum() / n



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for k, param in model.named_parameters():

        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            print(k)
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )