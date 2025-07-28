%Homework 1
%This assignment requires a descriptive statistical analysis of the NASA airfoil self-noise dataset. You will load the data and compute the mean, variance, median, kurtosis, skewness, and range for all variables. The objective is to quantitatively summarize the central tendency, dispersion, and shape of the data's distributions.

%Homework 2
%This project requires you to parse, structure, and preprocess a dataset of 3D surface geometries and their corresponding deformations. You will then apply PCA Whitening to normalize the data and use the t-SNE algorithm to visualize the results for a specific time step. The final objective is to analyze the t-SNE plots to comment on how the data clusters based on geometry, temperature, and pressure.


%Homework 3
%This project requires you to predict the complex flow stress of Austenitic Stainless Steel, a task for which traditional models are inadequate. You will build, train, and compare a baseline fully-connected neural network against a Long Short-Term Memory (LSTM) network using the provided thermomechanical data. The final objective is to evaluate both models' predictive accuracy and deploy them to forecast stress for a new, non-linear displacement profile.

%Homework 4
%This assignment requires you to tackle 3D object classification and part segmentation using the real-world ScanObjectNN point cloud dataset. You will implement, train, and optimize two foundational deep learning frameworks, PointNet and the hierarchical PointNet++, to process this unstructured 3D data. The final objective is to quantitatively compare the models' performance and analyze the challenges of applying them to noisy, real-world scans.

%Course Project:
%This study investigates the application of Variational Autoencoders (VAEs) to the synthesis of 2D microstructure images for computational materials design. A custom data-loading pipeline was developed in PyTorch to handle binary microstructure images and to generate paired clean/noisy examples via three distinct perturbation models: salt-and-pepper flips, structured sinusoidal grid overlays, and frequency-domain masking using FFT. The resulting dataset feeds into a denoising VAE framework that learns latent representations robust to diverse noise patterns. We further refine samples by training a Latent Diffusion Model (LDM) in the learned latent space, demonstrating that the diffusion process improves generation fidelity. Multi-modal noise augmentation and latent-space diffusion jointly yield high-quality synthetic microstructure realizations
