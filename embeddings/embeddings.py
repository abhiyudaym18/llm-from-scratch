import numpy 
import torch
import torch.nn as nn
import sys
from os import path
from tokenizer.tokenizer import Simpletokenizer


class Embedding(nn.Module):
    def __init__(self,vocabsize,embedding_dim):
        super().__init__()
        self.matrix = nn.Embedding(vocabsize,embedding_dim)
        

    def lookup(self,word):
        
        return self.matrix(word)


class Attention(nn.Module):
     
     def __init__(self):
        super().__init__()
        pass
     def softmax(self, score):
                exp_score = torch.exp(score)
                exp_sum = torch.sum(exp_score)
                return exp_score/exp_sum
     
     def forward(self, embeddings):
        dim = embeddings.shape[1]
        scale = dim ** 0.5
        scores = torch.matmul(embeddings, embeddings.transpose(0, 1)) / scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, embeddings)
        return output 
class PositionalEncoding(nn.Module):
     def __init__(self, sequence_length, embedding_dim):
          super().__init__()
          self.matrix = nn.Embedding(sequence_length,embedding_dim)
     def forward(self,sequence_length):
          positions = torch.arange(sequence_length, device=self.matrix.weight.device)
          return self.matrix(positions)
     
        
if __name__ == "__main__":

    t = Simpletokenizer()
    text = t.train("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    encoding = t.encode("Abhiyuday Mishra space")

    vocabsize = len(t.token_to_id)

    embed = Embedding(vocabsize, 4)

    attention = Attention()

    val = attention.forward(embed.lookup(encoding))

    print(val)