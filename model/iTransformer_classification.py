import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
import time


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_classes = configs.num_classes
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{configs.devices}')
        else:
            self.device = torch.device('cpu')

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(
            configs.d_model, self.seq_len, bias=True
        )
        self.classifier = nn.Linear(
            self.seq_len * 20, self.num_classes
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, padding_mask):
        # print("shape of x_enc, x_enc_mask", x_enc.shape, x_mark_enc.shape)
        # x_enc: [B,seq_len,N]; x_enc_mask: [B,seq_len,4]
        if self.use_norm:
        # Replace zeros with NaNs
            x_enc_nan = x_enc.clone().detach()
            x_enc_nan[x_enc_nan == 0] = float('nan')

            # Compute the mean, ignoring NaNs
            means = torch.nanmean(x_enc_nan, dim=1, keepdim=True)

            # Create a mask of non-NaN values
            mask = torch.isfinite(x_enc_nan)

            # Compute the variance, ignoring NaNs
            diff = (x_enc_nan - means) * mask
            var = torch.sum(diff * diff, dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)

            # Compute the standard deviation
            stdev = torch.sqrt(var + 1e-5)

            # Normalize x_enc
            x_enc = (x_enc - means) / stdev

            # Replace NaNs with zeros
            x_enc[torch.isnan(x_enc)] = 0

        _, L, N = x_enc.shape  # B L N
        self.seq_len = L
        # print("N",N)
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)

        enc_out, padding_mask = self.enc_embedding(
            x_enc, x_mark_enc, padding_mask
        )  # covariates (e.g timestamp) can be also embedded as tokens
        # print("shape of enc_out after enc_embedding", enc_out.shape)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules

        enc_out, attns = self.encoder(enc_out, attn_mask=padding_mask)
        # print("shape of enc_out after encoder", enc_out.shape)

        # B N E -> B N S -> B S N

        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates

        # Flatten the output tensor to feed it into the classification layer
        dec_out = dec_out.reshape(dec_out.size(0), -1)
        dec_out = self.classifier(dec_out)
        # print("shape of dec_out after classifier", dec_out.shape)
        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (
        #         stdev[:, 0, :].unsqueeze(1).repeat(1, self.num_classes, 1)
        #     )
        #     dec_out = dec_out + (
        #         means[:, 0, :].unsqueeze(1).repeat(1, self.num_classes, 1)
        #     )
        # print("shape of dec_out after de-normalization", dec_out.shape)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, padding_mask=mask)
        dec_out = dec_out.squeeze(-1)
        if self.output_attention:
            return dec_out, attns
        # return dec_out[:, -self.num_classes:, :]  # [B, L, D]
        else:
            return dec_out

    def reset(self, new_seq_len):
        configs = self.configs
        self.enc_embedding = DataEmbedding_inverted(
            new_seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        ).to(self.device)
        self.projector = nn.Linear(
            configs.d_model, self.seq_len, bias=True
        ).to(self.device)
        self.classifier = nn.Linear(
            self.seq_len * 20, self.num_classes
        ).to(self.device)

    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #     if self.use_norm:
    #         means = x_enc.mean(1, keepdim=True).detach()
    #         x_enc = x_enc - means
    #         stdev = torch.sqrt(
    #             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
    #         )
    #         x_enc /= stdev

    #     _, _, N = x_enc.shape

    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)

    #     # Apply the classifier to the last output of the encoder
    #     class_scores = self.classifier(enc_out[:, -1, :])

    #     # Apply softmax to convert raw scores into probabilities
    #     class_probs = F.softmax(class_scores, dim=-1)

    #     return class_probs
