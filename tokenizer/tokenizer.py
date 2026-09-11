#Create a class called SimpleTokenizer
#Add an __init__ method with no parameters except self, create two empty dictionaries inside it called token_to_id and id_to_token and store them as attributes
#Add a method called train with parameters self and text, inside it get unique characters from text, sort them, loop over them with enumerate, store character to number in token_to_id and number to character in id_to_token
#Add a method called encode with parameters self and text, inside it loop over every character in text, look each one up in token_to_id, collect the numbers into a list and return it
#Add a method called decode with parameters self and ids, inside it loop over every number in ids, look each one up in id_to_token, collect the characters into a list, join them into a string and return it
#Create an object called tokenizer from SimpleTokenizer
#Call tokenizer.train() with the string "hello world"
#Call tokenizer.encode() with "hello" and print the result
#Call tokenizer.decode() with the result from step 8 and print it, confirm you get "hello" back

class Simpletokenizer: 
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text):
        unique_char = sorted(set(text))
        pair = list(enumerate(unique_char, start=1))
        self.token_to_id = {"UNK" : 0}
        self.id_to_token = {0 : "UNK"}
        
        
        for (index, character) in pair:
            self.token_to_id[character] = index
            self.id_to_token[index] = character
       
    def encode(self, text):
        encoded = []
        for i in text:
            try:
                encoded.append(self.token_to_id[i])
            except:
                encoded.append(0)
        
        return encoded
    def decode(self, ids):
        decoded = []
        for i in ids:
            decoded.append(self.id_to_token[i])
        
        output = "".join(decoded)
        print(output)
        return output





#o1 = Simpletokenizer()
#o1.train("abcdefghijklmnopqrstuvwxyz")
#o1.encode("Hello%")
#val = [8, 5, 12, 12, 15]
#o1.decode(val)
