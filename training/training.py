from tokenizer.tokenizer import Simpletokenizer
from model.model import LLM
import torch 
import torch.nn as nn
import json



def  Train(text,epoch):
    tokenizer = Simpletokenizer()
    tokenizer.train(text)
    encoded = tokenizer.encode(text)
    vocabsize = len(tokenizer.token_to_id)
    embedding_dim=128
    llm_model = LLM(vocabsize, embedding_dim)
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
    torch.save(llm_model.state_dict(), 'model.pth')
    saved_model = {
        "Vocab_size" : vocabsize,
        "embedding_dim" : embedding_dim
    }
    with open("model_config_json", "w") as f:
        json.dump(saved_model,f)

    
if __name__ == "__main__":
    with open("moby_dick_or_the_whale.txt", "r", encoding="utf-8") as f:
        text = f.read()
    Train(text, epoch=1000)

    



