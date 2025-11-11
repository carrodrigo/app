# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class TabTransformerBinary(nn.Module):
    """
    Modelo TabTransformer adaptado para clasificación binaria.
    Permite el entrenamiento y predicción en datos tabulares mixtos (categóricos + numéricos).
    """
    def __init__(self, categories, num_continuous, dim=48, depth=4, heads=4,
                 attn_dropout=0.2, ff_dropout=0.2, mlp_hidden_mults=(4, 2)):
        super().__init__()

        # Definir el modelo base TabTransformer
        self.tabtransformer = TabTransformer(
            categories=categories,              # Lista con el número de clases únicas por variable categórica
            num_continuous=num_continuous,      # Número de variables numéricas
            dim=dim,                            # Dimensión de los embeddings
            depth=depth,                        # Capas del transformer
            heads=heads,                        # Número de cabezas de atención
            attn_dropout=attn_dropout,          # Dropout en la atención
            ff_dropout=ff_dropout,              # Dropout en la red feed-forward
            mlp_hidden_mults=mlp_hidden_mults   # Multiplicadores del tamaño del MLP final
        )

        # Capa de salida binaria
        self.fc_out = nn.Linear(dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, categorical, continuous=None, labels=None):
        """
        categorical: tensor [batch_size, n_cat]
        continuous: tensor [batch_size, n_cont]
        labels: tensor opcional [batch_size]
        """
        x = self.tabtransformer(categorical, continuous)
        logits = self.fc_out(x).squeeze(1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}

    def predict_proba(self, categorical, continuous=None):
        """Devuelve las probabilidades de clase positiva."""
        self.eval()
        with torch.no_grad():
            logits = self.fc_out(self.tabtransformer(categorical, continuous)).squeeze(1)
            probs = torch.sigmoid(logits)
        return torch.stack([1 - probs, probs], dim=1).cpu().numpy()

    def predict(self, categorical, continuous=None):
        """Devuelve la clase predicha (0 o 1)."""
        probs = self.predict_proba(categorical, continuous)
        return (probs[:, 1] >= 0.5).astype(int)


def build_model(categories, num_continuous):
    """
    Constructor auxiliar que devuelve un modelo con los hiperparámetros preconfigurados
    para el programa de Administración.
    """
    return TabTransformerBinary(
        categories=categories,
        num_continuous=num_continuous,
        dim=48,
        depth=4,           # Capas más ligeras para menor sobreajuste
        heads=4,           # Menor número de cabezas
        attn_dropout=0.2,  # Regularización moderada
        ff_dropout=0.2,
        mlp_hidden_mults=(4, 2)
    )
