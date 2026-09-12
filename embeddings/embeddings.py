import numpy 
import torch
import torch.nn as nn
import sys
from os import path
from tokenizer.tokenizer import Simpletokenizer


class Embedding(nn.Module):
    def __init__(self,vocabsize,embedding_dim):
        super().__init__()
        self.matrix = torch.rand(vocabsize,embedding_dim)
        

    def lookup(self,word):
        
        return self.matrix[word]


class Attention(nn.Module):
     
     def __init__(self):
        super().__init__()
        pass
     def softmax(self, score):
                exp_score = torch.exp(score)
                exp_sum = torch.sum(exp_score)
                return exp_score/exp_sum
     
     def forward(self,embeddings):
        num = len(embeddings)
        dim = embeddings.shape[1]
        scores = torch.zeros((num,num))
        weight = torch.zeros((num,num))
        output = torch.zeros((num, dim))

        for i in range(num):
            for j in range(num):
                scores[i][j] = torch.dot(embeddings[i], embeddings[j])
        for i in range(num):
            weight[i] = self.softmax(scores[i]) 
        for i in range(num):
             output[i] = torch.matmul(weight[i], embeddings)

        return output
    
if __name__ == "__main__":

    t = Simpletokenizer()
    text = t.train("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    encoding = t.encode("Abhiyuday Mishra space")

    vocabsize = len(t.token_to_id)

    embed = Embedding(vocabsize, 4)

    attention = Attention()

    val = attention.forward(embed.lookup(encoding))

    print(val)