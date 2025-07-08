
---

# DeepCNN_Pytorch

This repository demonstrates building deep convolutional neural network (DeepCNN) layers completely from scratch in PyTorch—without leveraging the built-in nn.Module or high-level abstractions. All components, including layers and forward/backward passes, are implemented manually for educational purposes.

## Project Overview

- **Goal:** Implement and train a deep neural network from scratch using only PyTorch tensors and operations, without using `nn.Module` or any high-level neural network utilities.
- **Dataset:** Breast cancer dataset with shape (144, 30) — 144 samples, each with 30 features.
- **Architecture Used:**
  - **Input:** 30 features
  - **Hidden Layer 1:** 4 neurons
  - **Hidden Layer 2:** 3 neurons
  - **Hidden Layer 3:** 2 neurons
  - **Output Layer:** 1 neuron, sigmoid activation

**X:    [N, 30]**
**↓**
**z1:   [N, 4]    ← W1: [30, 4]**
**a1:   [N, 4]**
**↓**
**z2:   [N, 3]    ← W2: [4, 3]**
**a2:   [N, 3]**
**↓**
**z3:   [N, 2]    ← W3: [3, 2]**
**a3:   [N, 2]**
**↓**
**z4:   [N, 1]    ← W4: [2, 1]**
**y_pred: [N, 1] → sigmoid(z4)**

   **    Input Layer (30 units)
  **       |
         V
 **W1: (30 x 4)  --> Hidden Layer 1 (4 neurons)**
         |
         V
  **W2: (4 x 3)   --> Hidden Layer 2 (3 neurons)**
         |
         V
  **W3: (3 x 2)   --> Hidden Layer 3 (2 neurons)**
         |
         V
   **W4: (2 x 1)   --> Output Layer (1 neuron, sigmoid)**






## Key Highlights

- **No Use of `nn.Module`**: All neural network components are hard-coded, providing hands-on understanding of the underlying mechanics.
- **Manual Forward and Backward Passes:** From weight initialization to backpropagation, every aspect is implemented from scratch.
- **Educational Focus:** Ideal for those wanting to learn how neural networks work under the hood, beyond auto-differentiation and high-level APIs.

## Results

- **Accuracy Achieved:** 62% (without any hyperparameter tuning, using raw learning).

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/malakazlan/DeepCNN_Pytorch.git
   cd DeepCNN_Pytorch
   ```

2. Open and run the Jupyter Notebook to see code, explanations, and results.

## Requirements

- Python 3.x
- PyTorch
- Jupyter Notebook
- Numpy

Install dependencies using:
```bash
pip install torch numpy notebook
```

## File Structure

- `DeepCNN_from_scratch.ipynb` — Main notebook containing all code and explanations.
- `README.md` — This file.

## Acknowledgements

- Breast cancer dataset used for demonstration and testing.


Feel free to edit or add more details as needed! If you want this as a markdown file or want it updated with additional info, let me know.
