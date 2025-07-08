# 🔬 Deep Neural Network from Scratch with PyTorch – Breast Cancer Classification

This project implements a **deep feedforward neural network from scratch using PyTorch**, without using high-level modules like `torch.nn.Module` or `torch.optim`. The model classifies breast cancer cases (malignant vs. benign) using the **Breast Cancer Wisconsin Diagnostic Dataset** from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

---

## 📌 Project Highlights

- ✅ Pure PyTorch (no `nn.Module`, no `optim`)
- ✅ Manual forward pass, backward pass, and parameter updates
- ✅ Supports any architecture (you define layer sizes)
- ✅ Trained on real-world breast cancer dataset
- ✅ Visualized loss & accuracy over epochs

---

## 🧠 Model Architecture

The neural network used in this project is **fully connected**, with the following architecture:

Input Layer : 30 features (after preprocessing)
Hidden Layer 1 : 4 neurons + ReLU
Hidden Layer 2 : 3 neurons + ReLU
Hidden Layer 3 : 2 neurons + ReLU
Output Layer : 1 neuron + Sigmoid

yaml
Copy
Edit

> Loss function: Binary Cross-Entropy (BCE)  
> Activation: ReLU (hidden), Sigmoid (output)  
> Optimizer: Manual gradient descent using `loss.backward()` and weight updates

---

## 📊 Dataset Overview

- **Source**: Kaggle – Breast Cancer Wisconsin (Diagnostic) Data Set  
- **Link**: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### Preprocessing:

- ✅ Dropped non-informative columns (`id`)
- ✅ Encoded target: `M` → 1, `B` → 0
- ✅ Scaled features using **StandardScaler** (mean=0, std=1)
- ✅ Converted data to PyTorch tensors (`dtype=torch.float64`)

---

## 📈 Training Results

- Final Accuracy: ~61%  
- Final Loss: ~0.72  
- Plotted **loss vs epochs** and **accuracy vs epochs**  
- Trained for 100 epochs with learning rate 0.01

> Note: Since this is a pure-from-scratch implementation, there’s room for optimization by tuning learning rate, epochs, adding regularization, or using better architectures.

---

## 📦 File Structure

├── deep_nn_from_scratch.py # All model, training, and evaluation logic
├── breast_cancer.csv # Dataset (from Kaggle)
├── README.md # This file
├── requirements.txt # Minimal dependencies

yaml
Copy
Edit

---

## 📌 Key PyTorch Concepts Practiced

- `requires_grad=True`
- `.backward()` for autograd
- Manual parameter updates with `torch.no_grad()`
- Broadcasting and shape alignment
- ReLU and Sigmoid activation functions
- Manual loss function implementation (Binary Cross-Entropy)

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install torch matplotlib pandas scikit-learn
2. Run the training script
bash
Copy
Edit
python deep_nn_from_scratch.py
📊 Sample Output
yaml
Copy
Edit
Epoch 0 | Loss: 0.6932 | Accuracy: 0.56
Epoch 10 | Loss: 0.6821 | Accuracy: 0.61
...
Epoch 99 | Loss: 0.7234 | Accuracy: 0.61
