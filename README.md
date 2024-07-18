## FashionGAN: Generating Fashion Images with Generative Adversarial Networks

This repository contains a Jupyter Notebook (GANs.ipynb) showcasing the implementation of a Generative Adversarial Network (GAN) for generating new fashion images. The model utilizes the Fashion-MNIST dataset, which consists of 28x28 grayscale images of clothing and accessories. 

**Table of Contents:**
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
    * [Generator](#generator)
    * [Discriminator](#discriminator)
* [Training](#training)
    * [Loss Functions & Optimizers](#loss-functions--optimizers)
    * [Training Loop](#training-loop)
    * [Model Monitoring](#model-monitoring)
* [Results](#results)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Future Work](#future-work)

## Introduction

Generative Adversarial Networks (GANs) are a powerful class of neural networks capable of learning complex data distributions and generating new, realistic data points. They consist of two competing networks:

* **Generator:** Learns to generate fake data samples that resemble the real data.
* **Discriminator:**  Learns to distinguish between real and generated (fake) data samples.

The generator and discriminator are trained simultaneously in an adversarial manner. The generator aims to fool the discriminator by generating increasingly realistic fake data, while the discriminator tries to improve its ability to identify the fakes. This adversarial training process eventually leads to a generator capable of producing high-quality, realistic data samples.

## Dataset

This project uses the Fashion-MNIST dataset, a popular benchmark for image classification and generation tasks. The dataset contains 70,000 grayscale images (28x28 pixels) categorized into 10 classes:

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

The dataset is pre-processed by scaling the pixel values from the range [0, 255] to [0, 1].

## Model Architecture

### Generator

The generator network takes a random noise vector as input and transforms it into a 28x28 grayscale image. It consists of a series of upsampling layers interspersed with convolutional layers.

* **Dense Layer:** Initially transforms the random noise vector into a higher-dimensional representation.
* **Reshape Layer:** Reshapes the output of the dense layer to form a low-resolution image.
* **Upsampling Blocks:** Progressively increase the resolution of the image using `UpSampling2D` layers followed by `Conv2D` layers.
* **Convolutional Blocks:** Extract features and refine the image details using `Conv2D` layers with LeakyReLU activations.
* **Output Layer:** A final `Conv2D` layer with a sigmoid activation function outputs a single-channel image representing the generated fashion item.

### Discriminator

The discriminator network takes a 28x28 grayscale image as input and outputs a probability score indicating whether the input is real or fake. 

* **Convolutional Blocks:** Extract features from the input image using `Conv2D` layers with LeakyReLU activations and dropout for regularization.
* **Flatten Layer:** Flattens the output of the convolutional layers into a 1D vector.
* **Dropout Layer:** Further regularizes the network to prevent overfitting.
* **Output Layer:** A dense layer with a sigmoid activation function outputs the probability score.

## Training

### Loss Functions & Optimizers

* **Generator Loss:** Utilizes Binary Cross-Entropy loss and aims to minimize the difference between the discriminator's output for generated images and a target of all ones (indicating real images).
* **Discriminator Loss:**  Also employs Binary Cross-Entropy loss and aims to minimize the difference between its output for real images and a target of all ones, and its output for generated images and a target of all zeros.

Both the generator and discriminator are trained using the Adam optimizer.

### Training Loop

The training loop iterates through the Fashion-MNIST dataset, feeding batches of real images and generated images to the discriminator and generator. The gradients from the loss functions are used to update the weights of both networks. 

### Model Monitoring

A custom callback (`ModelMonitor`) is implemented to monitor the training progress and generate sample images from the generator at the end of each epoch. These images provide a visual representation of the generator's progress in learning the data distribution.

## Results

After training, the generator successfully generates new fashion images that resemble the real data distribution. The quality of the generated images improves over epochs, showcasing the effectiveness of the GAN architecture in learning complex data distributions. 

## Getting Started

### Prerequisites

* Python 3.7+
* TensorFlow 2.x
* TensorFlow Datasets
* Matplotlib
* Jupyter Notebook (optional)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/FashionGAN.git
cd FashionGAN
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. Open and run the `GANs.ipynb` notebook in a Jupyter environment or execute the Python script.

2. The notebook guides you through the data loading, model building, training, and result visualization steps.

## Future Work

* **Improve Image Quality:** Explore advanced GAN architectures like DCGAN, WGAN, and StyleGAN to further improve the quality and diversity of generated images.
* **Conditional Image Generation:** Implement conditional GANs to generate images based on specific input conditions like clothing category, color, or style.
* **Image Inpainting/Editing:** Leverage GANs for tasks like image inpainting (filling in missing parts of an image) or editing (modifying specific aspects of an image).
