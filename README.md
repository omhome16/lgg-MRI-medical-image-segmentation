# 🎯 Automated Segmentation of Low-Grade Glioma (LGG) Tumors using U-NET and Attention U-NET

This project presents a deep learning pipeline for the **automated segmentation of LGG brain tumors** from **multi-modal MRI scans**, utilizing **U-NET** and **Attention U-NET** architectures. We compare both models through rigorous training, evaluation, and hyperparameter tuning to determine the most effective approach.

---

## 🧠 Project Highlights

- 📚 **Dataset**: BraTS LGG MRI Segmentation Dataset (T1, T1ce, T2, FLAIR modalities)
- 🧮 **Models**: U-NET and Attention U-NET
- 🧪 **Loss Function**: BCE + Dice Loss (weighted)
- ⚙️ **Optimizers**: Adam, AdamW
- 📉 **Schedulers**: StepLR, CosineAnnealingLR
- 📊 **Metric**: Dice Score for segmentation accuracy
- 🧵 **Best Test Dice Score**:
  - U-NET: **0.8726**
  - Attention U-NET: **0.6738**

---

🏗️ Model Architectures
U-NET:

      Parameters: 17.26M

      Achieved Test Dice: 0.8726 (after 50 epochs, tuned LR/scheduler)

Attention U-NET:

      Parameters: 17.61M

      Achieved Test Dice: 0.6738 (after 20 epochs)

⚙️ Best Hyperparameters
Model	LR	Optimizer	Scheduler	Epochs	Dice Score
U-NET	1e-4	Adam	Cosine	50	0.8726
Attention U-NET	1e-3	Adam	StepLR	20	0.6738

📌 Best setting found via exhaustive search over 24 combinations.

---


## 📂 Project Structure

```bash
├── data/                        # Preprocessed LGG MRI dataset
├── models/                     # U-NET and Attention U-NET architectures
├── scripts/
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Utilities for data loading, metrics, etc.
├── results/                    # Saved models, metrics, logs
├── HP_tuning.txt               # Hyperparameter tuning logs
├── requirements.txt            # Project dependencies
├── flow_UNET.txt               # U-NET training flow log
├── flow_ATTENTION_UNET.txt     # Attention U-NET training flow log
├── table.txt                   # Summary of experimental results
└── README.md

