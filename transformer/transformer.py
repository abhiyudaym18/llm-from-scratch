import torch
import torch.nn as nn
import sys
from os import path
from embeddings.embeddings import Attention

class FeedForword(nn.Module):

    def __init__(self,embedding_dim):
        super().__init__()
        self.linear_layer_1 = nn.Linear(embedding_dim,embedding_dim*4)
        self.relu = nn.ReLU()
        self.linear_layer_2 = nn.Linear(embedding_dim*4, embedding_dim)

    def Forward(self,data):
        x = self.linear_layer_1(data)
        x = self.relu(x)
        x = self.linear_layer_2(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.attention = Attention()
        self.feedforword = FeedForword(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self,x):
        x = x + self.attention(x)
        x = self.norm1(x)
        x = x + self.feedforword(x)
        x = self.norm2(x)
        return x

obj = TransformerBlock(4)

input_var = torch.randn(9,4)

obj.forward(input_var)
