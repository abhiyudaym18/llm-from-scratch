import torch
from model.model import LLM
from tokenizer.tokenizer import Simpletokenizer
import json


with open("model_config_json", "r") as f:
    saved_data = json.load(f)
def generate(model, prompt, num_tokens):
    
    generated_text = prompt

    for i in range(num_tokens):

        output = model.forward(generated_text)

        last_token_score = output[-1]

        next_token_id = torch.argmax(last_token_score)
        token_id = next_token_id.item()
        if token_id in model.tokenizer.id_to_token:
            next_char = model.tokenizer.id_to_token[token_id]
        else:
            next_char = 'UNK'

    

        generated_text = generated_text + next_char

    return generated_text

if __name__ == "__main__":
    vocab_size = saved_data.get("Vocab_size")
    embedding_dim = saved_data.get("embedding_dim")
    trained_model = LLM(vocab_size, embedding_dim)
    trained_model.tokenizer.train("Hello world this is my model")
    trained_model.load_state_dict(torch.load("model.pth"))
    trained_model.eval()
    while True: 
        prompt = input("Enter your prompt")
        print(generate(trained_model, prompt, num_tokens=20))