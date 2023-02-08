import torch.nn as nn
import torch
import numpy as np


class Clustering(nn.Module):
    def __init__(self, *, device, num_clusters=5, l_k, d_model):
        super(Clustering, self).__init__()

        self.device = device
        self.num_clusters = num_clusters
        self.proj_to_cluster = nn.Linear(l_k*d_model, num_clusters)
        self.q_proj = nn.Linear(num_clusters, num_clusters)
        self.k_proj = nn.Linear(num_clusters, num_clusters)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, Q, K):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        cluster_k = K.reshape(b, l_k*d_k*h)
        cluster_k_proj = self.proj_to_cluster(cluster_k)

        cluster_q = torch.softmax(self.q_proj(cluster_k_proj), dim=-1)
        cluster_k = torch.softmax(self.k_proj(cluster_k_proj), dim=-1)

        mu = torch.mean(cluster_q, dim=0)
        sigma = nn.Softplus()(torch.std(cluster_k, dim=0))

        dist = torch.distributions.normal.Normal(mu, sigma)
        likelihood = dist.log_prob(cluster_k)
        loss = -torch.mean(likelihood) + self.cross_entropy(cluster_q, cluster_q)

        inds = torch.argmax(cluster_q, dim=-1)
        inds = inds.unsqueeze(-1).repeat(1, l_k)
        inds = inds.unsqueeze(-1)

        K_re = K.reshape(b, l_k, -1)

        K_inds = torch.cat([K_re, inds], dim=-1)

        cluster_means = torch.zeros((self.num_clusters, l_k, d_k*h), device=self.device)

        K_cluster = K_inds[:, :, -1].unsqueeze(-1).repeat(1, 1, 1+d_k*h)

        for i in range(self.num_clusters):
            group = torch.where(K_cluster != i+1, K_inds, 0.0)
            cluster_means[i] = torch.mean(group[:, :, :-1], dim=0)[0]

        cluster_means = cluster_means.squeeze(1)

        cluster_means = cluster_means.unsqueeze(0).repeat(b, 1, 1, 1)

        cluster_means = cluster_means.reshape(b, h, self.num_clusters, l_k, d_k)

        scores_qk = torch.einsum('bhqd, bhkd-> bhqk', Q, K) / np.sqrt(d_k)
        scores_q_centers = torch.einsum('bhqd, bhckd-> bhcqk', Q, cluster_means) / np.sqrt(d_k)

        scores_qk = scores_qk.unsqueeze(2)

        final_scores = torch.cat([scores_qk, scores_q_centers], dim=2)

        final_score = torch.max(final_scores, dim=2)[0]

        return final_score, loss
