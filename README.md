### **README.md**
# GCCA Algorithm Reading Report

## Overview

This repository contains my study and reproduction experiments for the paper:  
**"Scalable and Flexible Multiview MAX-VAR Canonical Correlation Analysis"**  
by **Xiao Fu, Kejun Huang, Mingyi Hong, Nicholas D. Sidiropoulos, and Anthony Man-Cho So**.

The main goal is to **understand, implement, and experiment** with the **MaxVar-GCCA** algorithm presented in the paper.

---

## 📖 Paper Summary
- The paper introduces **MaxVar-GCCA**, an improved **Generalized Canonical Correlation Analysis (GCCA)** method.
- The method aims to **optimize large-scale multiview learning** by reducing computational complexity.
- The dataset used in the paper is a **word co-occurrence PMI matrix**, extracted from a multilingual corpus.

---

## 🛠️ Project Structure

<!-- ```plaintext
📂 GCCA-Algorithm-Reading-Report
│── 📂 datasets/           # PMI-based multilingual dataset (ignored in .gitignore)
│── 📂 src/                # Implementation of MaxVar-GCCA
│   ├── cca.py            # Standard CCA implementation
│   ├── gcca.py           # GCCA baseline implementation
│   ├── maxvar_gcca.py    # MaxVar-GCCA implementation (from paper)
│   ├── utils.py          # Helper functions
│── 📂 notebooks/          # Jupyter Notebooks for experiments
│── 📂 results/            # Saved experimental results and figures
│── .gitignore            # Ignore non-relevant files (e.g., datasets, cache)
│── README.md             # Project documentation (this file)
│── requirements.txt      # Dependencies for running the experiments
``` -->

## 🔥 Experiments & Findings

### **1️⃣ CCA & GCCA Implementation**
- Implemented **CCA, GCCA**, and **MaxVar-GCCA** from scratch.
- Compared **correlation performance** and **computational efficiency**.

### **2️⃣ PMI-based Multilingual Dataset**
- Reconstructed the dataset using **word co-occurrence matrices**.
- Verified that the dataset structure aligns with the paper’s description.

### **3️⃣ Evaluation of MaxVar-GCCA**
- Compared **MaxVar-GCCA vs. GCCA vs. CCA** on real & synthetic data.
- Measured **computation time, accuracy, and robustness**.

---

## 🚀 How to Use

### **🔧 Install dependencies**
```bash
pip install -r requirements.txt
```

### **📊 Run Experiments**
```bash
python src/maxvar_gcca.py --dataset datasets/multiview_pmi.pkl
```

### **📔 Open Jupyter Notebooks**
```bash
jupyter notebook
```
---

## 📌 Future Work
- Apply **MaxVar-GCCA** to **new datasets** (e.g., multilingual embeddings).
- Optimize **scalability** using sparse matrix methods.
- Experiment with **different feature extraction techniques**.

---

## 📜 References
1. **Original Paper**: [IEEE Link]([https://ieeexplore.ieee.org/document/XXXXXX](https://arxiv.org/abs/1605.09459))
2. **Dataset Source**: [PMI Matrix](https://sites.google.com/a/umn.edu/huang663/research) (currently unavailable)
3. **Related Work**: Generalized Canonical Correlation Analysis (GCCA) - A Review

---

## 🤝 Contributing
- If you're also studying **GCCA / MaxVar-GCCA**, feel free to open **issues** or contribute.
- For questions, reach out via **GitHub Issues**.

---

## 🏆 Acknowledgments
Thanks to **Xiao Fu, Kejun Huang, and co-authors** for their research work on GCCA. This project is my effort to better understand and reproduce their findings.

