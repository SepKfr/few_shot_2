import math

import torch.nn as nn
import torch
import numpy as np


class Clustering(nn.Module):
    def __init__(self, *, device, num_clusters=3, l, l_k, d_model):
        super(Clustering, self).__init__()

        self.device = device
        self.num_clusters = num_clusters

        log_l_k = int(math.log(l_k))
        log_l = int(math.log(l))

        self.shrink_k = nn.Linear(l_k, log_l_k, device=self.device)
        self.shrink_v = nn.Linear(l_k, log_l_k, device=self.device)
        self.shrink_q = nn.Linear(l, log_l, device=self.device)

        self.proj_to_cluster_k = nn.Sequential(nn.Linear(log_l_k*d_model, num_clusters, device=self.device),
                                             nn.ReLU())
        self.proj_to_cluster_q = nn.Sequential(nn.Linear(log_l * d_model, num_clusters, device=self.device),
                                               nn.ReLU())
        self.q_proj = nn.Linear(num_clusters, num_clusters, device=self.device)
        self.k_proj = nn.Linear(num_clusters, num_clusters, device=self.device)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, Q, K, V):

        b, h, l, d_k = Q.shape

        K = self.shrink_k(K.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        V = self.shrink_v(V.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        q_shrink = self.shrink_q(Q.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        l_k = K.shape[2]
        l_shrink = q_shrink.shape[2]

        cluster_k = K.reshape(b, l_k*d_k*h)
        cluster_q = q_shrink.reshape(b, l_shrink*d_k*h)
        cluster_k_proj = self.proj_to_cluster_k(cluster_k)
        cluster_q_proj = self.proj_to_cluster_q(cluster_q)

        cluster_q = torch.softmax(self.q_proj(cluster_q_proj), dim=-1)
        cluster_k = torch.softmax(self.k_proj(cluster_k_proj), dim=-1)

        mu = torch.mean(cluster_q, dim=0)
        sigma = nn.Softplus()(torch.std(cluster_k, dim=0))

        dist = torch.distributions.normal.Normal(mu, sigma)
        likelihood = dist.log_prob(cluster_k)
        loss = -torch.mean(likelihood) + self.cross_entropy(cluster_q, cluster_q)

        def get_ind_cluster(cluster, tup):

            seq_len = tup.shape[2]
            inds = torch.argmax(cluster, dim=-1)
            inds = inds.unsqueeze(-1).repeat(1, seq_len)
            inds = inds.unsqueeze(-1)

            tup = tup.reshape(b, seq_len, -1)

            tup_inds = torch.cat([tup, inds], dim=-1)

            tup_cluster = tup_inds[:, :, -1].unsqueeze(-1).repeat(1, 1, 1+d_k*h)
            tup_cluster = tup_cluster.long()

            return tup_cluster, tup_inds

        def get_group(cluster, inds, k):

            group = torch.where(cluster != k + 1, inds, 0.0)

            group = group[:, :, :-1]

            group = group.reshape(b, h, -1, d_k)

            return group

        scores_center = torch.zeros((self.num_clusters, b, h, l, l_k), device=self.device)

        Q_cluster, Q_inds = get_ind_cluster(cluster_q, Q)

        K_cluster, K_inds = get_ind_cluster(cluster_k, K)

        for i in range(self.num_clusters):

            group_Q = get_group(Q_cluster, Q_inds, i)
            group_K = get_group(K_cluster, K_inds, i)

            group_K = group_K.unsqueeze(2).repeat(1, 1, l_k, 1, 1)

            shape = [b, h, l_k, l, l_k]
            mask = np.tril(np.ones(shape))

            scores_q_group = torch.einsum('bhqd, bhckd-> bhcqk', group_Q, group_K) / np.sqrt(d_k)

            attn_mask = torch.as_tensor(mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores_q_group.masked_fill_(attn_mask, -1e9)

            scores_q_group = torch.softmax(scores_q_group, -1)

            scores_q_group = torch.mean(scores_q_group, dim=2)[0]

            q_ind = Q_cluster[:, :, :-1].reshape(b, h, -1, d_k)

            scores_center[i, torch.arange(b)[:, None, None],
                             torch.arange(h)[None, :, None],
            q_ind[:, :, :, 0], :] = scores_q_group

        final_score = torch.max(scores_center, dim=0)[0]
        attn = torch.softmax(final_score, -1)

        context = torch.einsum('bhqk, bhkd -> bhqd', attn, V)

        return context, loss
