from tokenizer.tokenizer import Simpletokenizer
from model.model import LLM
import torch 
import torch.nn as nn



def  Train(text,epoch):
    tokenizer = Simpletokenizer()
    tokenizer.train(text)
    encoded = tokenizer.encode(text)
    vocabsize = len(tokenizer.token_to_id)
    llm_model = LLM(vocabsize, embedding_dim=4)
    llm_model.tokenizer.train(text)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(llm_model.parameters(), lr=0.001)

    ids = encoded
    ids_tensor = torch.tensor(ids)

    for epoch in range(epoch):
        input = ids_tensor[:-1]
        target = ids_tensor[1:]

        output = llm_model.forward(text)
        output = output[:-1]

        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch%100 == 0:
            print(epoch)
            print(loss)

if __name__ == "__main__":
    Train("hello world this is a simple language model", epoch=1000)
    


    



