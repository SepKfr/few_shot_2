import numpy as np
import torch
import torch.nn as nn
import random
from modules.clustering import Clustering

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class BasicAttn(nn.Module):

    def __init__(self, d_k, h, device, seed, l_k, few_shot):

        super(BasicAttn, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device
        self.d_k = d_k

        self.few_shot = few_shot
        if self.few_shot:
            self.cluster = Clustering(device=device, l_k=l_k, d_model=d_k*h)

        self.layer_norm = nn.LayerNorm(d_k, device=self.device)

    def forward(self, Q, K, V, attn_mask):

        if self.few_shot:

            cntx, loss = self.cluster(Q, K, V)
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
            attn = torch.softmax(scores, -1)
            context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
            context_f = self.layer_norm(context + cntx)

            return [context_f, loss]

        else:
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
            attn = torch.softmax(scores, -1)
            context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
            return [context, 0.0]
