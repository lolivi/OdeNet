# OdeNet
PyTorch implementation of a Implicit Neural Representation with Ordinary Differential Equations used to find speed and direction of meteor tracks (Source: https://arxiv.org/abs/2308.14948)

# Mini-EUSO
Mini- EUSO is a wide-field-of-view imaging telescope that operates onboard the International Space Station since 2019 collecting data on miscellaneous processes that take place in the atmosphere of Earth in the UV range. Meteors are among these events and they can be observed leaving a linear track in the Mini-EUSO Field of View. 

# Implicit Neural Representation 
Implicit neural representations, also referred to as coordinate-based representations, are neural networks used to parameterize continuous and differentiable signals, such as images. In this way,the signal is encoded in the neural network parameters and it is often the only way an image could be parameterized, as an analytical function would be impossible to derive. Source: https://arxiv.org/abs/2006.09661

# Physics-Based Model
The physics-based model is inspired from https://arxiv.org/abs/2204.14030. 
The neural network performs a regression by mapping a pixel coordinate to the Mini-EUSO video sequences of meteor tracks. The video lasts 10 frames, and the resolution is 11 × 11 pixels, with the meteor track starting at the center. 
The physics dynamics has been implemented in the neural network with a timespatial transformation 𝑇 acting on two separate implicit neural representations. 
- The global pixel coordinates (𝑋, 𝑌) are used as input for the first neural representation, returning the background photon counts.
- Then, the pixel coordinates are mapped by function 𝑇 to local reference frame (𝑥, 𝑦) of the meteor through its kinematics. The meteor starts moving in the first frame at pixel (𝑋0, 𝑌0) at time 𝑡 = 0. In the following frames, 𝑡 = 1, ..., 10, the meteor position is determined by its apparent speed 𝑣 in x-axis and y-axis: 
-- 𝑇 (𝑋0, 𝑌0, 𝑣, 𝜃, 𝑡) = (𝑋0 + 𝑣 cos(𝜃) 𝑡, 𝑌0 + 𝑣 sin(𝜃) 𝑡)
-- (𝑥, 𝑦) = (𝑋, 𝑌) − 𝑇 (𝑋0, 𝑌0, 𝑣, 𝜃, 𝑡)
- The local pixel depends on meteor dynamics and it is used as input in the second implicit neural representation, making the model based on the physics of the signal. The output of the second neural representation represents the signal itself and is summed over the background photon counts. 

# Python scrits
- odenet_pytorch.py #pytorch architecture and libraries
- odenet_tuning.py #hyperparameter tuning
- odenet_training.py #training of whole dataset
- odenet_analysis.py #plot after training
