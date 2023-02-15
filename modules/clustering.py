import math

import numpy
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class Clustering(nn.Module):
    def __init__(self, *, device, num_clusters=5, d_model):
        super(Clustering, self).__init__()

        self.device = device
        self.num_clusters = num_clusters

        self.proj_to_cluster_k = nn.Sequential(nn.Linear(d_model, num_clusters,
                                                         device=self.device),
                                                         nn.ReLU())
        self.proj_back_to_cluster_k = nn.Sequential(nn.Linear(num_clusters, d_model,
                                                              device=self.device),
                                                              nn.ReLU())
        self.cluster_k_proj = nn.Linear(num_clusters, num_clusters, device=self.device)
        self.cluster_q_proj = nn.Linear(num_clusters, num_clusters, device=self.device)

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, Q, K, V):

        b, h, l, d_k = Q.shape

        K = nn.MaxPool1d(kernel_size=9, padding=int((9-1)/2))(K.reshape(b, d_k*h, -1)).reshape(b, h, -1, d_k)
        V = nn.MaxPool1d(kernel_size=9, padding=int((9-1)/2))(V.reshape(b, d_k*h, -1)).reshape(b, h, -1, d_k)

        l_k = K.shape[2]

        padding = torch.zeros_like(K)
        K_padded = torch.cat([padding, K[1:]])
        K_unfold = K_padded.unfold(0, int(b/2), 1)

        K_unfold = K_unfold.reshape(b, l_k, -1, d_k*h)

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

        ind_clusters = ind_clusters.unsqueeze(-1).repeat(1, 1, 1, self.num_clusters)

        cluster_centers = [torch.mean(cluster_q.clone().masked_fill_((ind_clusters == i), 0.0), dim=2)
                           for i in range(self.num_clusters)]

        cluster_center = torch.stack(cluster_centers)

        cluster_center = self.proj_back_to_cluster_k(cluster_center).reshape(b, h, self.num_clusters, l_k, d_k)
        scores_center = torch.einsum('bhqd, bhckd -> bhqk', Q, cluster_center)

        attn = torch.softmax(scores_center, -1)

        context = torch.einsum('bhqk, bhkd -> bhqd', attn, V)

        return context, loss
