import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Elects_layer import ClassificationHead, DecisionHead
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.num_classes = configs.num_classes
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
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
        self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        self.classification_head = ClassificationHead(9, self.num_classes)
        self.stopping_decision_head = DecisionHead(9)

        # self.classifier = nn.Linear(81, self.num_classes)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print("shape of x_enc, x_enc_mask", x_enc.shape, x_mark_enc.shape)
        # x_enc: [B,seq_len,N]; x_enc_mask: [B,seq_len,4]
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # print("N",N)
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(
            x_enc, x_mark_enc
        )  # covariates (e.g timestamp) can be also embedded as tokens

        # print("shape of enc_out after enc_embedding", enc_out.shape)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print("shape of enc_out after encoder", enc_out.shape)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates

        # Flatten the output tensor to feed it into the classification layer
        log_class_probabilities = self.classification_head(dec_out)
        probabilitiy_stopping = self.stopping_decision_head(
            dec_out
        )  # print("shape of dec_out after classifier", dec_out.shape)

        # print("shape of dec_out after de-normalization", dec_out.shape)
        return log_class_probabilities, probabilitiy_stopping

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        log_class_probabilities, probabilitiy_stopping = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )

        # print("dec out shape", dec_out.shape)
        # print("dec out", dec_out[:, -self.num_classes:, :].shape, dec_out[:, -self.num_classes:, :])
        # dec_out = self.fc(dec_out)
        # dec_out = F.softmax(dec_out, dim=-1)
        # print("dec out", dec_out.shape, dec_out)
        # return dec_out[:, -self.num_classes:, :]  # [B, L, D]
        return log_class_probabilities, probabilitiy_stopping

    @torch.no_grad()
    def predict(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):
        logprobabilities, deltas = self.forward(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )

        def sample_stop_decision(delta):
            dist = torch.stack([1 - delta, delta], dim=1)
            return torch.distributions.Categorical(dist).sample().bool()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t])
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(tuple(stop_now.shape)).bool()
                if torch.cuda.is_available():
                    last_stop = last_stop.to(device)
                stop.append(last_stop)

        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).bool()

        # is only true if stopped for the first time
        first_stops = (stopped.cumsum(1) == 1) & stopped

        # time of stopping
        t_stop = first_stops.long().argmax(1)

        # all predictions
        predictions = logprobabilities.argmax(-1)
        print(f"{predictions=}")
        print(f"{first_stops=}")
        print(f"{predictions.shape=}")
        print(f"{first_stops.shape=}")

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return logprobabilities, deltas, predictions_at_t_stop, t_stop
