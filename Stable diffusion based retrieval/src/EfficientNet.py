from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from torch import nn
import torch
import numpy as np
from tqdm import tqdm


torch.manual_seed(42)

weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
model = efficientnet_v2_l(weights=weights)
model = nn.Sequential(*list(model.children())[:-1]).to("cuda:0")  # Delete last layer
model.eval()

N_BATCH = 3


def get_embeddings(images):
    preprocess = weights.transforms()
    if len(images) > 1:
        batch = torch.stack([preprocess(i) for i in images])
    else:
        batch = preprocess(images[0]).unsqueeze(0)

    batch = batch.to("cuda:0")

    result = None
    total_iter = int(np.ceil(len(batch) / N_BATCH))
    for i in tqdm(range(0, total_iter), total=total_iter):
        pred = np.array([embedding.flatten().detach().cpu().numpy() for embedding in model(batch[i*N_BATCH:i*N_BATCH+N_BATCH])])
        result = np.concatenate((pred, result)) if result is not None else pred

    return result
