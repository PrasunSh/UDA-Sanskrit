# Domain Separation Network (DSN) for Speech Recognition

This repository implements the **Domain Separation Network (DSN)** for **unsupervised domain adaptation (UDA)** in **Automatic Speech Recognition (ASR)**.  
It is based on the paper *"Unsupervised Domain Adaptation Schemes for Building ASR in Low-Resource Languages"* and is tailored for **Hindi → Sanskrit** adaptation with **frame-level senone labels**.

---

## 📌 Features
- **Lazy loading** of per-utterance `.npy` features (saves memory).
- **Collate function** for variable-length speech sequences.
- **Private & shared encoders** for source/target domains.
- **Decoder & reconstruction loss** for feature consistency.
- **Senone classifier** (source-only supervision).
- **Domain classifier** for adversarial adaptation.
- Supports **unsupervised training** on target domain (no labels).
- End-to-end training loop with **classification, domain, difference, and reconstruction losses**.
- **WER** and **CER** evaluation on test set.

---

## 📂 Repository Structure
```
.
├── dsn_pipeline.ipynb   # Main notebook (training + evaluation)
├── README.md            # Project documentation
├── data/
│   ├── hindi/           # Source features (.npy per utterance)
│   ├── senone_labels/   # Matching senone labels (.npy per utterance)
│   ├── sanskrit_train/  # Target train features (.npy per utterance)
│   └── sanskrit_test/   # Target test features (.npy per utterance)
└── transcripts/
    └── filtered_transcripts.txt  # Reference transcripts for evaluation
```

---

## ⚙️ Requirements
Install dependencies before running:
```bash
pip install torch numpy jiwer editdistance
```

---

## 🚀 Training Pipeline

### 1. Data Preparation
- Place `.npy` feature files per utterance in `data/hindi/` and `data/sanskrit_*`.
- Place matching senone label `.npy` files in `data/senone_labels/`.
- Ensure filenames match between features and labels (e.g., `utt1.npy` ↔ `utt1.npy`).

### 2. Model Components
- **Private Encoder (source/target):** domain-specific representations.  
- **Shared Encoder:** domain-invariant features.  
- **Decoder:** reconstructs original input from private + shared.  
- **Senone Classifier:** predicts frame-level senones (source only).  
- **Domain Classifier:** predicts domain (source vs. target).  

### 3. Loss Functions
- **L_cls (CE loss):** classification loss on source senones.  
- **L_domain:** adversarial loss for domain discrimination.  
- **L_diff:** orthogonality constraint between private & shared.  
- **L_recon:** reconstruction consistency.  

**Total Loss:**
```
L_total = L_cls + α * L_domain + β * L_diff + γ * L_recon
```

---

## 🏋️ Training Example
```python
path1 = r"C:\npy_feats\hindi"
path2 = r"C:\senone_labels"
sanskrit_train = r"C:\npy_feats\sanskrit_train"
sanskrit_test  = r"C:\npy_feats\sanskrit_test"

src_loader, tgt_loader, tgt_test_loader = get_dataloaders(
    path1, path2, sanskrit_train, sanskrit_test, batch_size=16
)

dsn_model = DSN(input_dim=1320, num_senones=3080).to(device)
train_dsn(dsn_model, src_loader, tgt_loader, num_epochs=20)
```

Example training log:
```
Epoch 01/20 | Avg loss: 8.0693
Epoch 05/20 | Avg loss: 2.5964
Epoch 10/20 | Avg loss: 2.1740
Epoch 20/20 | Avg loss: 1.8976
```

---

## 📊 Evaluation (WER & CER)
Requires a reference transcript file (`filtered_transcripts.txt`) in the format:
```
utt1.m4a|reference transcription of utterance 1
utt2.m4a|reference transcription of utterance 2
```

Run evaluation:
```python
from jiwer import wer

print(f"WER: {wer_score:.3f}")
print(f"CER: {cer_score:.3f}")
```

Example output:
```
WER: 0.000
CER: 0.000
```

*(Note: WER/CER=0.0 indicates a mismatch in alignment or incorrect decoding step. Ensure transcripts and predictions are mapped correctly.)*

---

## 🔮 Next Steps
- Integrate a **CTC or seq2seq decoder** for real transcript-level predictions.  
- Improve evaluation pipeline beyond senone IDs.  
- Experiment with **gradient reversal layer (GRL)** for domain classifier stability.  
- Extend to other low-resource languages.  

---

## 📜 References
- Bousmalis et al., *Domain Separation Networks*, NeurIPS 2016.  
- Shahnawazuddin et al., *Unsupervised Domain Adaptation Schemes for Building ASR in Low-Resource Languages*.  
