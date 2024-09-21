import torch as t
import numpy as np
from torch import nn

class Model(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    
    def forward(self, x):
        return self.model(x)
    
def predict(model, embedding_model, text):
    embedding = t.tensor(embedding_model.encode(text))
    logits = model(embedding)
    return t.argmax(logits).item()