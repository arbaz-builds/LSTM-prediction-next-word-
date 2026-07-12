"""FastAPI server for the LSTM text generator — tiny footprint version.

Uses a small character-level vocabulary (not tiktoken's ~100k tokens) so the
model is small enough to run comfortably on free-tier (512MB RAM) hosting.
No pre-trained checkpoint is used — weights are randomly initialized, so
output is not coherent text. This is intentional, for deployment/demo
purposes only.
"""
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

device = torch.device("cpu")

# Tiny fixed character vocabulary instead of tiktoken's ~100k tokens.
_CHARS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-\n"
VOCAB_SIZE = len(_CHARS)
CHAR_TO_ID = {c: i for i, c in enumerate(_CHARS)}
ID_TO_CHAR = {i: c for i, c in enumerate(_CHARS)}
UNK_ID = 0
SEQ_LENGTH = 128


def encode(text: str) -> list:
    return [CHAR_TO_ID.get(c, UNK_ID) for c in text]


def decode(ids: list) -> str:
    return "".join(ID_TO_CHAR.get(i, "") for i in ids)


class LSTMTextGenerator(nn.Module):
    """Same architecture as train.py, but tiny (char-level) vocab + hidden size."""
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)


app = FastAPI(
    title="LSTM Next-Word Prediction API (tiny demo build)",
    description="From-scratch PyTorch LSTM text generator, deployed with a small "
                 "character-level vocabulary and no pre-trained checkpoint — "
                 "output is from randomly-initialized weights, for deployment demo purposes.",
    version="1.0.0",
)

_model = LSTMTextGenerator(vocab_size=VOCAB_SIZE)
_model.eval()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 40
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    output: str
    model_status: str


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device), "vocab_size": VOCAB_SIZE}


@app.post("/generate", response_model=GenerateResponse, summary="Generate text from a prompt")
async def generate_endpoint(payload: GenerateRequest):
    if not payload.prompt.strip():
        return GenerateResponse(output="", model_status="untrained_random_weights")

    with torch.no_grad():
        tokens = encode(payload.prompt)[-SEQ_LENGTH:]
        for _ in range(payload.max_tokens):
            inp = torch.tensor([tokens[-SEQ_LENGTH:]])
            logits = _model(inp)[:, -1, :] / payload.temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            tokens.append(next_tok)

    return GenerateResponse(output=decode(tokens), model_status="untrained_random_weights")
