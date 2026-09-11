import numpy 
import sys
from os import path
from tokenizer.tokenizer import Simpletokenizer
class Embedding:
    def __init__(self,vocabsize,embedding_dim):
        self.matrix = numpy.random.rand(vocabsize,embedding_dim)
        

    def lookup(self,word):
        
        return self.matrix[word]
t = Simpletokenizer()
text = t.train("abcdefghijklmnopqrstuvwxyz")
encoding = t.encode("Hello World")

vocabsize = len(t.token_to_id)

embed = Embedding(vocabsize, 4)

print(embed.lookup(encoding))