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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm_model = llm_model.to(device)
    ids = encoded
    ids_tensor = torch.tensor(ids)
    ids_tensor = ids_tensor.to(device)
    llm_model.tokenizer.train(text)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(llm_model.parameters(), lr=0.0003)

    

    sequence_length = 128
    print(f"Training on: {device}")
    print(f"Vocab size: {vocabsize}")
    print(f"Total tokens: {len(ids_tensor)}")
    print("Starting training...")

    for epoch in range(epoch):
        start = torch.randint(0, len(ids_tensor) - sequence_length, (1,)).item()
        
        input_chunk = ids_tensor[start:start + sequence_length]
        target_chunk = ids_tensor[start + 1:start + sequence_length + 1]
        
        output = llm_model.forward_ids(input_chunk)
        
        loss = loss_func(output, target_chunk)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
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
    Train(text, epoch=50000)

    



