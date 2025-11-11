# model_tabtransformer.py
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class TabTransformerBinary(nn.Module):
    def __init__(self, categories, num_continuous, dim=32, depth=4, heads=8,
                 attn_dropout=0.1, ff_dropout=0.1, mlp_hidden_mults=(4, 2)):
        super().__init__()

        self.tabtransformer = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            mlp_hidden_mults=mlp_hidden_mults
        )

        self.fc_out = None
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, categorical, continuous, labels=None):
        x = self.tabtransformer(categorical, continuous)
        if self.fc_out is None:
            out_features = x.shape[1]
            self.fc_out = nn.Linear(out_features, 2).to(x.device)
        logits = self.fc_out(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.long())
        return {"loss": loss, "logits": logits}
