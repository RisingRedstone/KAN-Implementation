# Kolmogorov–Arnold Networks (KAN) – TensorFlow Implementation

🚀 A from-scratch implementation of **Kolmogorov–Arnold Networks (KANs)** in TensorFlow, inspired by the Kolmogorov–Arnold representation theorem.  
This project explores **spline-based neural architectures** that offer **interpretability, efficient function approximation, and model compression** compared to standard MLPs.

---

## ✨ Features

- **Custom KAN Layers**: Implemented B-spline basis functions combined with SiLU activations.  
- **Dynamic Grid Updates**: Functions to re-parameterize spline grids (`gridChange`, `UpdateGrid`) for flexible approximation.  
- **Regularization & Pruning**: Adaptive loss terms and neuron pruning to compress models while retaining accuracy.  
- **Custom Training Loop**: Low-level gradient tape training with NaN-safe gradient correction and learning rate scheduling.  
- **Visualization Tools**: Functions to visualize neuron activity, spline weights, and interpret learned representations.  
- **Benchmarking**: Trained on **MNIST dataset**, achieving >96% accuracy with pruning and compression.  

---

## 📊 Results

- Achieved **96% accuracy on MNIST** with fewer parameters compared to equivalent MLPs.  
- Pruning significantly reduced redundant neurons while maintaining performance.  
- Visualization showed interpretable spline activations across input domains.  

---

## 🛠 Tech Stack

- **Language**: Python  
- **Frameworks**: TensorFlow, NumPy, Matplotlib  
- **Concepts**: Neural Networks, Spline Interpolation, Model Compression, Gradient-Based Optimization  

---
