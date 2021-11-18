# Objectives
This project is to investigate the effects of hyperparameters such as initial learning rate, learning rate schedule, weight decay, and data augmentation on deep neural networks.
One of the most important issues in deep learning is optimization versus regularization. Optimization is controlled by the initial learning rate and the learning rate schedule. Regularization is controlled by, among other things, weight decay and data augmentation. As a result, the values of these hyperparameters are absolutely critical for the performance of deep neural networks.

# Dataset and Network
## CIFAR-10
The CIFAR-10 dataset consists of 60000 32 × 32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## ResNet-18
ResNet-18 contains two different kind of residual blocks, the regular residual block, and the residual block with 1 × 1 convolution. With residual blocks, inputs can forward propagate faster through the residual connections. Downsampling is removed from the originial ResNet, because the image size in CIFAR-10 is 32 × 32.  

# Experiment
The most important thing is to get the priorities right. There are too many hyperparameters to perform exhaustive tuning. Many do not have strong influence on the result. First, achieve close-to-zero training loss (learning rate, momentum, optimizer). After that, increase regularization to reduce overfitting.

## Learning rate
The batch size is set to 128. Neither weight decay nor learning rate schedule is used. Run three experiments with the learning rate set to 0.1, 0.01, and 0.001 respectively. Train the networks for 15 epochs under each setting.

## Learning Rate Schedule
When we adjust learning rate, we look for one that minimizes training loss. Using this criterion, $\eta=0.01$ is identified as the best learning rate the last experiment. Use this as the initial learning rate and keep other hyperparameters unchanged. Conduct experiments under two settings: (1) train for 300 epochs with the learning rate held constant, and (2) train for 300 epochs with cosine annealing.

## Weight Decay
Instead of gradient descent on $\mathcal{L}'(w)$, we can perform gradient descent on $\mathcal{L}(w)$ and subtract $\eta\lambda w$ from the current $w$ in each update. Directly applying the subtraction on $w$ is called weight decay. Surprisingly, weight decay often outperforms L2 regularization.
Add weight decay to the experimental settings used previously (including the best learning rate and the cosine schedule). Experiment with two different weight decay coefficients $\lambda$, $5 \times 10^{−4}$ and $1\times 10^{-2}$,

## Data Augmentation
Up to now, we have found the best learning rate, the cosine schedule, and weight decay. Finally, we apply cutout augmentation technique to the dataset for further reducing overfitting. 