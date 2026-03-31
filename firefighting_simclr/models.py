from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# Use the resnet18 architecture as the backbone for both SimCLR and classification models.
def build_backbone(name: str = "resnet18") -> tuple[nn.Module, int]:
    if name != "resnet18":
        raise ValueError(f"Unsupported backbone: {name}")

    backbone = models.resnet18(weights=None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim

# after the encoder backbone, 
# we add a projection head for SimCLR that maps the features to a lower-dimensional space for contrastive learning,
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# consist of the encoder backbone followed by a projection head
class SimCLRModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        projection_hidden_dim: int = 256,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(feature_dim, projection_hidden_dim, projection_dim)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_a = self.encoder(view_a)
        hidden_b = self.encoder(view_b)
        projection_a = self.projector(hidden_a)
        projection_b = self.projector(hidden_b)
        return projection_a, projection_b

# for the finetuning phase, we add a classification head on top of the encoder backbone,
# not using the projection head from SimCLR, since the classification task may require different features than the contrastive learning task.
class ClassificationModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        return self.classifier(features)

# the normalized temperature-scaled cross-entropy loss (NT-Xent) for contrastive learning
# as indicated in the SimCLR paper, 
# which encourages the projections of positive pairs to be similar 
# while pushing apart the projections of negative pairs in the feature space.
def nt_xent_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size = projection_a.size(0)
    projections = torch.cat([projection_a, projection_b], dim=0)
    projections = F.normalize(projections, dim=1)

    similarity = projections @ projections.T
    similarity = similarity / temperature

    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, torch.finfo(similarity.dtype).min)

    positive_indices = torch.arange(batch_size, device=similarity.device)
    targets = torch.cat([positive_indices + batch_size, positive_indices], dim=0)

    return F.cross_entropy(similarity, targets)


def load_encoder_state_dict(checkpoint_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)

    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if encoder_state:
        return encoder_state

    return state_dict
