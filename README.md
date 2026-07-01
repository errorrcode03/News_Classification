# 📰 News Category Classification

A production-quality **Deep Learning** project that classifies BBC News articles into 5 categories using a **Bidirectional LSTM** with **GloVe 100d embeddings**, built entirely in Jupyter Notebooks.

---

## 🏗️ Project Structure

```
News_Classification/
│
├── notebooks/
│   ├── news_classification.ipynb  ← Full end-to-end notebook (main)
│   ├── train.ipynb                ← Standalone training notebook
│   ├── evaluate.ipynb             ← Evaluation + confusion matrix
│   └── predict.ipynb              ← Inference + confidence charts
│
├── data/
│   ├── bbc-text.csv               ← Auto-downloaded BBC dataset
│   └── glove/
│       └── glove.6B.100d.txt      ← Auto-downloaded GloVe vectors
│
├── models/
│   ├── word2idx.pkl               ← Saved vocabulary
│   ├── label_encoder.pkl          ← Saved label mappings
│   ├── config_snapshot.json       ← Config used during training
│   ├── confusion_matrix.png       ← Confusion matrix plot
│   └── training_curves.png        ← Loss & accuracy plots
│
├── app.py                         ← Streamlit web application
├── saved_model.pth                ← Best trained model weights
├── config.yaml                    ← Hyperparameter configuration
├── requirements.txt               ← Python dependencies
└── README.md
```

---

## 🎯 Categories

| Category | BBC Label |
|---|---|
| Business | `business` |
| Entertainment | `entertainment` |
| Politics | `politics` |
| Sports | `sport` |
| Technology | `tech` |

---

## 🧠 Model Architecture

```
Embedding Layer (GloVe 100d, fine-tuneable)
        ↓
Bidirectional LSTM (hidden=128, layers=2, dropout=0.5)
        ↓
Dropout (0.5)
        ↓
Linear (256 → 128)  +  ReLU
        ↓
Dropout (0.3)
        ↓
Linear (128 → 5)
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding dim | 100 |
| Hidden size | 128 |
| LSTM layers | 2 |
| Bidirectional | ✅ |
| LSTM dropout | 0.5 |
| FC dropout | 0.3 |
| Batch size | 64 |
| Epochs | up to 20 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | CosineAnnealingLR |
| Grad clipping | 1.0 |
| Early stopping | patience=5 |

---

## ⚙️ Setup (Conda)

### 1. Create & activate the Conda environment

```bash
conda create -n news_classification python=3.11 -y
conda activate news_classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly streamlit pyyaml tqdm jupyter notebook
```

### 3. Register the Conda kernel with Jupyter

```bash
python -m ipykernel install --user --name news_classification --display-name "news_classification"
```

---

## 🚀 Usage

### Option A — Full end-to-end notebook

```bash
jupyter notebook notebooks/news_classification.ipynb
```

Run all 15 cells in order. This handles everything: dataset download, GloVe download, preprocessing, training, evaluation, and saving artifacts.

### Option B — Individual notebooks

| Notebook | Purpose |
|---|---|
| `train.ipynb` | Train the model from scratch |
| `evaluate.ipynb` | Evaluate on test set |
| `predict.ipynb` | Classify your own articles |

```bash
jupyter notebook
```

### Option C — Streamlit Web App

After training (running `train.ipynb` or `news_classification.ipynb`):

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Dataset

- **Source:** [BBC News dataset](https://huggingface.co/datasets/SetFit/bbc-news) via HuggingFace
- **Size:** 2,225 articles
- **Split:** 80% train / 10% val / 10% test (stratified)
- **Auto-downloaded** inside the notebook on first run

---

## 📦 GloVe Embeddings

- **Source:** Stanford NLP — GloVe 6B 100d
- **Size:** ~822 MB (zip)
- **Auto-downloaded** inside the notebook on first run
- Saved to `data/glove/glove.6B.100d.txt`

---

## 📈 Expected Results

| Metric | Expected Value |
|---|---|
| Test Accuracy | ~96–98% |
| Macro Precision | ~96–98% |
| Macro Recall | ~96–98% |
| Macro F1 | ~96–98% |

---

## 🌐 Streamlit App Features

- 🎨 Dark glassmorphism UI
- ✍️ Text input for any article
- ⚡ Quick-fill example buttons
- 🎯 Predicted category with icon badge
- 📊 Interactive Plotly confidence bar chart
- 📐 Token count metrics
- 🧠 Sidebar with model architecture details

---

## 🔮 Future Improvements

- [ ] BERT / DistilBERT transformer model
- [ ] Multi-label classification
- [ ] REST API with FastAPI
- [ ] Docker containerisation
- [ ] More categories (world news, science, health)
- [ ] Attention mechanism visualisation
- [ ] Model quantisation for faster inference

---

## 📋 Reproducibility

All random seeds are fixed at **42** across Python, NumPy, and PyTorch for reproducible results.

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.
