import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class Clustering(nn.Module):
    def __init__(self, *, device, num_clusters=5, batch_size, l_k):
        super(Clustering, self).__init__()

        self.device = device
        self.num_clusters = num_clusters

        log_l_k = int(np.log(l_k))
        self.shrink_k = nn.Linear(l_k, 2*log_l_k, device=self.device)
        self.shrink_v = nn.Linear(l_k, 2*log_l_k, device=self.device)

        self.proj_to_cluster_k = nn.Sequential(nn.Linear(batch_size, num_clusters, device=self.device),
                                               nn.ReLU())
        self.cluster_k_proj = nn.Linear(num_clusters, num_clusters, device=self.device)
        self.cluster_q_proj = nn.Linear(num_clusters, num_clusters, device=self.device)

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, Q, K, V):

        b, h, l, d_k = Q.shape

        K = self.shrink_k(K.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        V = self.shrink_v(V.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        l_k = K.shape[2]

        padding = torch.zeros_like(K)
        K_padded = torch.cat([padding, K[1:]])
        K_unfold = K_padded.unfold(0, b, 1)

        cluster_k_p = self.proj_to_cluster_k(K_unfold)

        cluster_k = self.cluster_k_proj(cluster_k_p)
        cluster_q = self.cluster_q_proj(cluster_k_p)

        cluster_k = torch.softmax(cluster_k, dim=-1)
        cluster_q = torch.softmax(cluster_q, dim=-1)

        mu = torch.mean(cluster_q, dim=-1)
        sigma = nn.Softplus()(torch.std(cluster_q, dim=-1))

        dist = torch.distributions.normal.Normal(mu, sigma)
        likelihood = dist.log_prob(torch.mean(cluster_k, dim=-1))
        loss = -torch.mean(likelihood) + self.cross_entropy(mu, mu)

        ind_clusters = torch.argmax(cluster_q, dim=-1)
        ind_clusters = ind_clusters.long()
        ind_clusters = ind_clusters.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_clusters)

        cluster_center = torch.zeros(self.num_clusters, b, h, l_k, d_k, device=self.device)

        for i in range(self.num_clusters):

            cluster_q_sub = torch.where(ind_clusters == i, cluster_q, 0.0)
            cluster_q_sub = torch.mean(cluster_q_sub, dim=-1)
            cluster_center[i] = cluster_q_sub

        cluster_center = cluster_center.reshape(b, h, -1, l_k, d_k)
        scores_center = torch.einsum('bhqd, bhckd -> bhcqk', Q, cluster_center)

        final_score = torch.max(scores_center, dim=2)[0]
        attn = torch.softmax(final_score, -1)

        context = torch.einsum('bhqk, bhkd -> bhqd', attn, V)

        return context, loss
