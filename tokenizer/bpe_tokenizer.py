""""
Step 1 — __init__
Create two empty properties: self.vocab as a dict and self.merges as a list.

Step 2 — start of train
Take every unique character in text, sort them, assign each an ID starting from 0. Store in self.vocab. Same as your original tokenizer but store the string as key and ID as value.

Step 3 — split text into characters
tokens = list(text) — this gives you a list where every element is one character.

Step 4 — count pairs
Loop over tokens with range(len(tokens) - 1). On each iteration, create a tuple of (tokens[i], tokens[i+1]) and increment its count in a dictionary.

Step 5 — find best pair
Look up "python max dictionary by value" — one line.

Step 6 — merge
Create new_token = best_pair[0] + best_pair[1]. Add to vocab. Append to merges. Then loop over tokens and replace adjacent occurrences of the pair with the new token.

Step 7 — repeat
Wrap steps 4-6 in a while loop until vocab reaches vocab_size.
"""

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def train(self, text,vocab_size):
        unique_char = sorted(set(text))
        pair = list(enumerate(unique_char, start=1))
        self.vocab = {"UNK" : 0}
        for (index, character) in pair:
            self.vocab[character] = index
        tokens = list(text)
        count = {}
        for i in range (len(tokens) - 1):
            couple = (tokens[i], tokens[i+1])
            count[couple] = count.get(couple, 0) + 1
        best_pair = max(count, key=count.get)
        print(best_pair)
if __name__ == "__main__":
    with open("./moby_dick_or_the_whale.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = BPETokenizer()
    tokenizer.train(text[:100000], 300)