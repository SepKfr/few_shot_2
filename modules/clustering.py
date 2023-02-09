import torch.nn as nn
import torch
import numpy as np


class Clustering(nn.Module):
    def __init__(self, *, device, num_clusters=5, l_k, d_model):
        super(Clustering, self).__init__()

        self.device = device
        self.num_clusters = num_clusters
        self.proj_to_cluster = nn.Sequential(nn.Linear(l_k*d_model, l_k),
                                             nn.Linear(l_k, num_clusters),
                                             nn.ReLU())
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

        K_cluster = K_inds[:, :, -1].unsqueeze(-1).repeat(1, 1, 1+d_k*h)

        scores_center = torch.zeros((self.num_clusters, b, h, l, l_k), device=self.device)

        for i in range(self.num_clusters):

            group = torch.where(K_cluster != i+1, K_inds, 0.0)

            group = group[:, :, :-1]

            group = group.reshape(b, h, l_k, -1)

            shape = [b, h, l, l_k]
            mask = np.tril(np.ones(shape))

            scores_q_group = torch.einsum('bhqd, bhkd-> bhqk', Q, group) / np.sqrt(d_k)

            attn_mask = torch.as_tensor(mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores_q_group.masked_fill_(attn_mask, -1e9)

            scores_q_group = torch.softmax(scores_q_group, -1)

            scores_center[i] = scores_q_group

        scores_qk = torch.einsum('bhqd, bhkd-> bhqk', Q, K) / np.sqrt(d_k)

        scores_qk = scores_qk.unsqueeze(0)

        final_scores = torch.cat([scores_qk, scores_center], dim=0)

        final_score = torch.max(final_scores, dim=0)[0]

        return final_score, loss
