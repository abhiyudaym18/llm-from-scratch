import numpy 
import sys
from os import path
from tokenizer.tokenizer import Simpletokenizer


class Embedding:
    def __init__(self,vocabsize,embedding_dim):
        self.matrix = numpy.random.rand(vocabsize,embedding_dim)
        

    def lookup(self,word):
        
        return self.matrix[word]


class Attention:
     def __init__(self):
        pass
     def softmax(self, score):
                exp_score = numpy.exp(score)
                exp_sum = numpy.sum(exp_score)
                return exp_score/exp_sum
     
     def Forward(self,embeddings):
        num = len(embeddings)
        dim = embeddings.shape[1]
        scores = numpy.zeros((num,num))
        weight = numpy.zeros((num,num))
        output = numpy.zeros((num, dim))

        for i in range(num):
            for j in range(num):
                scores[i][j] = numpy.dot(embeddings[i], embeddings[j])
        for i in range(num):
            weight[i] = self.softmax(scores[i]) 
        for i in range(num):
             output[i] = numpy.dot(weight[i], embeddings)

        return output
    


t = Simpletokenizer()
text = t.train("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
encoding = t.encode("Abhiyuday")

vocabsize = len(t.token_to_id)

embed = Embedding(vocabsize, 4)

attention = Attention()

val = attention.Forward(embed.lookup(encoding))

print(val)