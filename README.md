# UCD MSc Advanced AI Projects Portfolio

Welcome to my University College Dublin **GitHub Portfolio** !

Here are gathered all the labs and projects I completed during my **MSc Advanced Artificial Intelligence** at UCD.

This repository contains Jupyter notebooks showcasing my initial hands-on implementations of learned concepts.

All of the projects on this portfolio are built based around some learning concepts of the UCD MSc AAI's modules.

## Objective

The objective of this portfolio is to:
- Demonstrate my technical skills in Data Science and AI
- Showcase my understanding of theoretical concepts
- Provide clear explanations of my methodologies and results
- Track my progress and learning throughout my MSc degree

## Projects

### Advanced Machine Learning

<br>

- Focused on advanced and state-of-the-art machine learning methods, including deep learning, ensemble models, unsupervised and semi-supervised learning, reinforcement learning, and human-in-the-loop systems.  
- Developed the ability to implement, and evaluate appropriate ML algorithms using Python-based toolkits.  
- Emphasized critical understanding of algorithmic design, performance evaluation, and societal impact.
<br>

1. [Lab1](https://github.com/axemar/UCD-Portfolio/tree/main/AdvancedMachineLearning) : Implementation of a **Random Forest Classifier** model on the **fashion-mnist** dataset, then use of a **Grid Search CV** to find the best parameters.

2. [Lab2](https://github.com/axemar/UCD-Portfolio/tree/main/AdvancedMachineLearning) : Implementation of the **Random**, **Greedy**, $\Large \varepsilon$**-Greedy**, $\Large \varepsilon$**-First** and $\Large \varepsilon$**-Decreasing** strategies for the Reinforcement Learning **$k$-armed** bandit policy experiment.

---

### Computer Vision And Imaging

<br>

- Focused on applying modern AI and deep learning techniques to vision and imaging tasks, including image classification, object detection, segmentation, and enhancement.  
- Developed hands-on experience with state-of-the-art architectures such as CNNs, Vision Transformers, and generative models (GANs, diffusion), using transfer learning and fine-tuning strategies.  
- Built the ability to evaluate vision models, interpret their behavior, and assess their strengths/limitations.
<br>

1. [Lab1](https://github.com/axemar/UCD-Portfolio/tree/main/ComputerVisionAndImaging) : Development of a **mini Vision Pipeline**: an image filter that applies a vintage sepia tone with vignetting, as well as a ‘Cool Blue Film’ style, combining **color shift, brightness adjustment, and a soft Gaussian blur**.

---

### Foundations Deep Learning

<br>

- Introduced the core principles of modern deep learning, covering neural network fundamentals.  
- Developed practical skills in designing, training, and evaluating deep learning models for computer vision and natural language processing using frameworks such as PyTorch and TensorFlow.  
- Explored advanced architectures including CNNs, RNNs, Transformers, and generative models (VAEs, GANs, diffusion), with a critical assessment of their strengths and limitations.

1. [Lab1](https://github.com/axemar/UCD-Portfolio/tree/main/FoundationsDeepLearning) : Creation of an **iterative algorithm** from **scratch** to learn how to discriminate images using a **logistic regression classification model**.

2. [Lab2](https://github.com/axemar/UCD-Portfolio/tree/main/FoundationsDeepLearning) : Creation of a **Single Layer Feedforward Neural Network** from **scratch** to learn highly **non-linear detection regions**.

---

### Optimisation

<br>

- Introduced core optimisation techniques, including convex optimisation, linear and integer programming.  
- Developed the ability to model and solve real-world optimisation problems (linear programming and integer linear programming) and to apply the basic optimisation techniques to solve machine learning problems.  
- Built a solid understanding of convex optimisation foundations underlying machine learning and deep learning algorithms, including gradient-based approaches and its variants.

$\newline$

1. [Lab1](https://github.com/axemar/UCD-Portfolio/tree/main/Optimisation) : Introduction to **Julia/JuMP** and use of the **GLPK solver** to model and solve **LP problems** following the simplex method.

2. [Lab2](https://github.com/axemar/UCD-Portfolio/tree/main/Optimisation) : Use of the **graphical method** (plot the constraints, the feasible space, and the optimal point) to solve **LP problems** following the simplex method.

3. [Lab3](https://github.com/axemar/UCD-Portfolio/tree/main/Optimisation) : Use of the **simplex tableaux method** to solve LP problems throught multiple iterations, highlighting **intermediate basis** and **feasible space**.

4. [Lab4](https://github.com/axemar/UCD-Portfolio/tree/main/Optimisation) : Use of the **duality theorem** (bounded, unbounded, infeasible) to solve **primal & dual** LP problems following the simplex method.

5. [Lab5](https://github.com/axemar/UCD-Portfolio/tree/main/Optimisation) : Formulation of the **binary integer program** to solve ILP problems following the **Branch & Bound** algorithm and the **LP Relaxation** method.

## Directory Structure

```bash
UCD-Portfolio
┣ AdvancedMachineLearning
┃ ┣ Lab1.ipynb
┃ ┗ Lab2.ipynb
┣ ComputerVisionAndImaging
┃ ┗ Lab1.ipynb
┣ FoundationsDeepLearning
┃ ┣ Lab1.ipynb
┃ ┗ Lab2.ipynb
┣ Optimisation
┃ ┣ Lab1.ipynb
┃ ┣ Lab2.ipynb
┃ ┣ Lab3.ipynb
┃ ┣ Lab4.ipynb
┃ ┗ Lab5.ipynb
┣ .gitignore
┣ LICENSE
┗ README.md
```

New files will be added soon.

## Notes

- The datasets, figs, `utils.py` files and `.pyc` files are not included due to size and/or licensing constraints
- The notebooks are shared to demonstrate my capabilities (not to ensure reproducibility)
- This portfolio is actively maintained and updated