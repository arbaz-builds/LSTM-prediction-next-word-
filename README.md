# LSTM Next-Word Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![tiktoken](https://img.shields.io/badge/Tokenizer-tiktoken%20cl100k-412991)](https://github.com/openai/tiktoken)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An LSTM-based language model built from scratch in PyTorch that predicts the next tokens given a text prompt — using the same tokenizer as GPT-4 (`tiktoken`, `cl100k_base`).

```
Input:  "The weather today is"
Output: "The weather today is sunny and warm"
```

---

## Architecture

```
Raw Text
   │
   ▼
tiktoken (cl100k_base)        ← same tokenizer as GPT-4, 100k+ vocab
   │
   ▼
Token Embeddings (256-dim)
   │
   ▼
LSTM Layer (512 hidden units, 1 layer)
   │
   ▼
FC(512→1024) → ReLU → Dropout(0.3) → FC(1024→512) → FC(512→vocab_size)
   │
   ▼
Softmax → Next Token
```

| Component | Detail |
|---|---|
| Tokenizer | `tiktoken` `cl100k_base` (GPT-4 style) |
| Embedding | 256-dim, learned |
| LSTM | 512 hidden units, 1 layer, batch-first |
| Classifier head | 3-layer FC with ReLU + Dropout(0.3) |
| Sequence length | 512 tokens (configurable) |
| Training data | Q&A pairs from `Dataset.csv` |
| Device | Auto-detects CUDA, falls back to CPU |

## Project structure

```
LSTM-prediction-next-word-/
├── train.py           # Training script — CLI-configurable dataset path & hyperparameters
├── inference.py        # Model loading + interactive text-generation CLI
├── Dataset.csv          # Q&A training data
├── requirements.txt     # Dependencies
└── README.md
```

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/arbaz-builds/LSTM-prediction-next-word-.git
cd LSTM-prediction-next-word-
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

Dataset path and hyperparameters are CLI-configurable:

```bash
python train.py --data /kaggle/input/your-dataset/Dataset.csv --epochs 20 --lr 0.0001
```

| Flag | Default | Purpose |
|---|---|---|
| `--data` | `Dataset.csv` | Path to the training CSV |
| `--epochs` | `15` | Number of training epochs |
| `--seq-length` | `512` | Training sequence length |
| `--lr` | `0.0001` | Learning rate |
| `--batch-size` | `32` | Batch size |

The best checkpoint (lowest training loss) is saved to `best_model.pth`.

### 3. Run inference

```bash
python inference.py
```

```
LSTM Text Generator
   Device  : cuda
   Vocab   : 100,277 tokens
   Type 'exit' to quit

Prompt: The weather today is
Output: The weather today is sunny and pleasant outside
```

## Key features

- Auto-detects CUDA, falls back to CPU — no manual device configuration needed
- GPT-4-compatible tokenizer (`tiktoken` `cl100k_base`, 100k+ vocab)
- CLI-configurable training — dataset path and hyperparameters don't require editing the script
- Temperature-controlled sampling at inference time
- Clear error handling — `inference.py` exits with a helpful message if `best_model.pth` is missing

## Known limitations

- **Single-layer LSTM** — no stacked layers or bidirectional variant yet
- **No beam search** — inference uses multinomial sampling only
- **No pretrained weights shipped** — the model must be trained locally before `inference.py` will run
- **Fixed sequence length at inference** — `inference.py` currently hardcodes `SEQ_LENGTH=512`; if you train with a different `--seq-length`, update this constant to match

## Roadmap

- [x] LSTM training from scratch
- [x] GPT-4-compatible tokenizer integration
- [x] CLI-configurable training script
- [ ] Pretrained model weights release
- [ ] Streamlit / Gradio demo UI
- [ ] Beam search decoding
- [ ] Multi-layer / bidirectional LSTM support

## Tech stack

| Library | Purpose |
|---|---|
| PyTorch | LSTM model definition & training loop |
| tiktoken | GPT-4-style tokenization |
| Pandas | Dataset loading & preprocessing |
| NumPy | Numerical operations |

## Author

**Mohammad Arbaz** — AI/ML Developer
[GitHub @arbaz-builds](https://github.com/arbaz-builds)

## License

MIT — see [LICENSE](LICENSE).
