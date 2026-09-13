import torch
import torch.nn as nn
from  tokenizer.tokenizer import Simpletokenizer 
from  embeddings.embeddings import Embedding, Attention
from transformer.transformer import TransformerBlock

class LLM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        #Inside it create: tokenizer, embedding, attention, transformer block, and output linear layer
        self.tokenizer = Simpletokenizer()
        self.embeddings = Embedding(vocab_size,embedding_dim)
        self.attention = Attention()
        self.transformerblock = TransformerBlock(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, text):
        ids = self.tokenizer.encode(text)
        ids_tensor = torch.tensor(ids)

        embeded = self.embeddings.lookup(ids_tensor)
        attentded = self.attention.forward(embeded)
        tranformer = self.transformerblock.forward(attentded)
        output = self.output(tranformer)

        return output
    
if __name__ == "__main__":
    t = Simpletokenizer()
    t.train("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    vocab_size = len(t.token_to_id)
    
    model = LLM(vocab_size, 4)
    model.tokenizer.train("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    output = model.forward("hello")
    print(output)
    print(output.shape)