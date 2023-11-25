from typing import List

from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7
from torch import nn
import torch
import numpy as np

torch.manual_seed(42)

weights = EfficientNet_B7_Weights.IMAGENET1K_V1
model = efficientnet_b7(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])  # Delete last layer

def get_embeddings(images):
    preprocess = weights.transforms()
    if len(images) > 1:
        batch = torch.stack([preprocess(i) for i in images])
    else:
        batch = preprocess(images[0]).unsqueeze(0)

    return np.array([embedding.flatten().detach().numpy() for embedding in model(batch)])
