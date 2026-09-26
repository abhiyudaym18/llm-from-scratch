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
        temperature = 0.8
        probs = torch.softmax(last_token_score/ temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
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
    with open("moby_dick_or_the_whale.txt", "r", encoding="utf-8") as f:
        training_text = f.read()
    trained_model.tokenizer.train(training_text)
    trained_model.load_state_dict(torch.load("model.pth"))
    trained_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)
    while True: 
        prompt = input("Enter your prompt")
        print(generate(trained_model, prompt, num_tokens=20))