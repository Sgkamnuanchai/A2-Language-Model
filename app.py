from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from Model.LSTMLanguageModel import LSTMLanguageModel
from datasets import load_from_disk
import pickle
import numpy as np
import random
import platform
app = Flask(__name__)

# load dataset
with open("./Model/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print("Vocabulary size:", len(vocab))

# define params for model
emb_dim = 1024                # 400 in the paper
hid_dim = 1024                # 1150 in the paper
num_layers = 2                # 3 in the paper
dropout_rate = 0.65 
vocab_size = len(vocab)
tokenizer = get_tokenizer('basic_english')
max_seq_len = 30
temperature = 0.5
seed = 0

# Device selection
if platform.system() == "Darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("macOS")
else:
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print("Windows or Linux")

# Load model and vocabulary
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('./Model/best-val-lstm_lm.pt',  map_location=device))

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    try:
        # Get JSON data from the POST request
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate text
        generated_tokens = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
        generated_text = " ".join(generated_tokens)

        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

