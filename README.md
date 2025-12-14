# Generative Models: GANs and VAEs

## M608 Business Project in Computer Science

A comprehensive exploration of Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) across multiple datasets and architectures, including a mobile application for model deployment.

## Project Overview

This project implements and compares various generative models across different datasets, focusing on:

- **Deep Convolutional GANs (DCGANs)** for image generation
- **Conditional GANs (CGANs)** for controlled generation
- **Variational Autoencoders (VAEs)** for latent space representation
- **Cross-platform deployment** through an iOS application

## Project Structure

### Datasets & Implementations

#### CIFAR-10

- **DCGAN** ([cifar_10/dcgan/main.py](cifar_10/dcgan/main.py))
  - Deep convolutional architecture for 32×32 color images
  - Multiple training runs with hyperparameter tuning
- **VAE** ([cifar_10/vae/](cifar_10/vae/))
  - Standard VAE implementation
  - SiLU activation variant for improved stability

#### MNIST

- **Digits DCGAN** ([mnist/digits/dcgan.py](mnist/digits/dcgan.py))
  - Classic handwritten digit generation
- **Fashion MNIST** ([mnist/fashion/](mnist/fashion/))
  - Conditional GAN for fashion items
  - VAE implementation for clothing generation

#### Quick Draw

- **VAE** ([quick_draw/vae/](quick_draw/vae/))
  - Sketch-based generative modeling

### Mobile Application

- **iOS App** ([GANs/](GANs/))
  - Swift-based mobile application
  - Real-time model inference on device
  - Integration of trained generator models

### Analysis & Experiments

- [analysis.ipynb](analysis.ipynb) - Model evaluation
- [DCGAN_MNIST.ipynb](DCGAN_MNIST.ipynb) - DCGAN experiments
- [ipynb/vae.ipynb](ipynb/vae.ipynb) - VAE analysis

### Model Artifacts

- [models_weights/](models_weights/) - Saved model checkpoints from training runs
  - SafeTensors format for efficient serialization
  - Multiple snapshots across training epochs
  - Generated image samples for evaluation

## Setup

### Python Environment

```bash
uv sync
```

### iOS Application

Open [GANs/GANs.xcodeproj](GANs/GANs.xcodeproj) in Xcode to build and run the mobile app.

## Key Features

### Implemented Architectures

- **DCGAN**: Stable training with convolutional architectures
- **Conditional GAN**: Label-conditioned generation for controlled outputs
- **VAE**: Latent space learning with reconstruction and KL divergence losses

### Training Optimizations

- Multiple output directories tracking training experiments
- Hyperparameter tuning across learning rates, batch sizes, and architectures
- Model checkpointing at regular intervals
- Loss tracking and visualization

### Mobile Deployment

- Swift-MLX integration for on-device inference
- Optimized model weights for mobile performance
- Real-time generation capabilities

## Project Goals

1. **Comparative Analysis**: Evaluate GANs vs VAEs for different generation tasks
2. **Architecture Exploration**: Test various network designs and activation functions
3. **Practical Deployment**: Deploy models in production-ready mobile application
4. **Dataset Diversity**: Train across multiple datasets to understand model generalization

## Training Results

Model weights and generated samples are preserved in the [models_weights/](models_weights/) directory, organized by timestamp. Each training run includes:

- Generator and discriminator/encoder-decoder weights
- Generated image samples

## Future Work

- Extend to additional datasets (CelebA, ImageNet subsets)
- Implement StyleGAN architectures
- Enhance mobile app
- Compare computational efficiency across architectures

---

**Course**: M608 Business Project in Computer Science  
**Focus**: Image Generative Models, Mobile Deployment
