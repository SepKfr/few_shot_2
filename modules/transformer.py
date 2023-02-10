import torch.nn as nn
import torch
import random
import numpy as np
from modules.encoder import Encoder
from modules.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_input_size, tgt_input_size, pred_len, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device, attn_type, seed, few_shot):
        super(Transformer, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.attn_type = attn_type

        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, attn_type=attn_type, seed=seed, few_shot=few_shot)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=1, pad_index=tgt_pad_index,
            device=device,
            attn_type=attn_type, seed=seed, few_shot=few_shot)

        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.projection = nn.Linear(d_model, 1, bias=False)
        self.attn_type = attn_type
        self.pred_len = pred_len
        self.device = device

        self.final_projection = nn.Linear(d_model, 1)
        self.few_shot = few_shot

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs = self.enc_embedding(enc_inputs)
        dec_outputs = self.dec_embedding(dec_inputs)

        if self.few_shot:
            enc_outputs, enc_loss = self.encoder(enc_outputs)
            dec_outputs, dec_loss, dec_enc_loss = self.decoder(dec_outputs, enc_outputs)
            loss_dec_enc = enc_loss + enc_loss + dec_enc_loss

        else:
            enc_outputs = self.encoder(enc_outputs)
            dec_outputs = self.decoder(dec_outputs, enc_outputs)

        outputs = self.final_projection(dec_outputs[:, -self.pred_len:, :])

        if self.few_shot:
            return outputs, loss_dec_enc
        else:
            return outputs