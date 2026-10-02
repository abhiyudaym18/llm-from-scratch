import torch
import torch.nn as nn
from  tokenizer.tokenizer import Simpletokenizer 
from  embeddings.embeddings import Embedding, Attention, PositionalEncoding
from transformer.transformer import TransformerBlock

class LLM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        #Inside it create: tokenizer, embedding, attention, transformer block, and output linear layer
        self.tokenizer = Simpletokenizer()
        self.embeddings = Embedding(vocab_size,embedding_dim)
        self.attention = Attention()
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim) for _ in range(4)])
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.positional_encoding = PositionalEncoding(512, embedding_dim)

    def forward(self, text):
        ids = self.tokenizer.encode(text)
        ids_tensor = torch.tensor(ids)
        ids_tensor = ids_tensor.to(next(self.parameters()).device)

        embeded = self.embeddings.lookup(ids_tensor)
        seq_len = embeded.shape[0]
        embeded = embeded + self.positional_encoding.forward(seq_len)
        attentded = self.attention.forward(embeded)
        x = attentded
        for block in self.blocks:
            x = block(x)
        output = self.output(x)

        return output
    def forward_ids(self, ids_tensor):
        embeded = self.embeddings.lookup(ids_tensor)
        seq_len = embeded.shape[0]
        embeded = embeded + self.positional_encoding.forward(seq_len)
        attended = self.attention.forward(embeded)
        x = attended
        for block in self.blocks:
            x = block(x)
        output = self.output(x)
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