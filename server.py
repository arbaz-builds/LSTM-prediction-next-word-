"""FastAPI server for the LSTM text generator.

Note: this deploys without a pre-trained best_model.pth checkpoint.
The model runs with randomly-initialized weights, so generated text
will be incoherent/random until a real trained checkpoint is added.
This is intentional for demo/deployment purposes.
"""
import os
import torch
import torch.nn as nn
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("cl100k_base")
VOCAB_SIZE = enc.n_vocab
SEQ_LENGTH = 512


class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out.reshape(-1, lstm_out.size(-1))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        logits = self.fc3(out)
        b, s, _ = lstm_out.shape
        return logits.view(b, s, -1)


app = FastAPI(
    title="LSTM Next-Word Prediction API",
    description="From-scratch PyTorch LSTM text generator. "
                 "Running without a trained checkpoint (best_model.pth not present) "
                 "— output is from randomly-initialized weights until a trained model is added.",
    version="1.0.0",
)

_model = None
_model_status = "not_loaded"


def _get_model() -> LSTMTextGenerator:
    global _model, _model_status
    if _model is None:
        _model = LSTMTextGenerator(vocab_size=VOCAB_SIZE)
        if os.path.exists("best_model.pth"):
            _model.load_state_dict(torch.load("best_model.pth", map_location=device))
            _model_status = "trained_checkpoint_loaded"
        else:
            _model_status = "untrained_random_weights"
        _model.to(device)
        _model.eval()
    return _model


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 40
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    output: str
    model_status: str


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


@app.post("/generate", response_model=GenerateResponse, summary="Generate text from a prompt")
async def generate_endpoint(payload: GenerateRequest):
    model = _get_model()
    if not payload.prompt.strip():
        return GenerateResponse(output="", model_status=_model_status)

    with torch.no_grad():
        tokens = enc.encode(payload.prompt)
        for _ in range(payload.max_tokens):
            inp = torch.tensor([tokens[-SEQ_LENGTH:]]).to(device)
            logits = model(inp)[:, -1, :] / payload.temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            tokens.append(next_tok)

    return GenerateResponse(output=enc.decode(tokens), model_status=_model_status)
