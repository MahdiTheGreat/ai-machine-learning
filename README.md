# Artificial Intelligence, Machine Learning & Data Science Suite

A comprehensive monorepo consolidating machine learning systems, deep learning architectures, computer vision pipelines, natural language processing models, and statistical data engineering projects.

---

## Technical Highlights & Domains

### 1. Computer Vision & Deep Learning
* **Land-Cover Aerial Semantic Segmentation ([`Semantic-Segmentation-with-Deep-Learning`](projects/Semantic-Segmentation-with-Deep-Learning/)):**  
  Trained U-Net and Encoder-Decoder architectures for 6-class land-cover semantic segmentation on high-resolution aerial imagery. Implemented targeted data augmentation and class-weighting strategies to mitigate severe spatial and class imbalance across geographic zones.
* **Handwritten Digit Classification ([`MNIST_pytorch_classification`](projects/MNIST_pytorch_classification/)):**  
  Convolutional Neural Network (CNN) built in PyTorch demonstrating baseline image recognition, backpropagation dynamics, and evaluation curves.

---

### 2. Conversational AI & Natural Language Processing
* **Extensible Multi-Domain Conversational Agent ([`AItools`](projects/AItools/)):**  
  Developed an intent-driven conversational agent with Rasa for weather, transit, and food queries. Built with a modular, extensible design facilitating custom entity extraction, pipeline slots, and rule-based action dispatching.
* **Text Classification & Filter Engine ([`Spam-Ham`](projects/Spam-Ham/)):**  
  Natural language filtering pipeline utilizing TF-IDF vectorization and probabilistic classification (Naive Bayes) to classify message corpora.

---

### 3. Recommender Systems
* **Hybrid Recommender Architecture ([`RecommenderSystem`](projects/RecommenderSystem/)):**  
  Engineered collaborative filtering (matrix factorization and user-item interactions) and content-based filtering models. Addressed the cold-start challenge for new users and sparse items using hybrid weighting strategies.

---

### 4. Predictive Modeling & Supervised Machine Learning
* **Kaggle Spaceship Titanic Classification ([`Spaceship_Titanic`](projects/Spaceship_Titanic/)):**  
  Binary classification pipeline predicting passenger transport status, achieving **80.1% accuracy** with a hyperparameter-tuned Random Forest model. Features advanced missing-data imputation pipelines, categorical encoding, and domain-specific feature engineering.
* **Clinical Diagnostics & Real Estate Regression ([`Diagnostic-Systems`](projects/Diagnostic-Systems/), [`DiabetesClassifier`](projects/DiabetesClassifier/), [`LifeExpectancy`](projects/LifeExpectancy/)):**  
  * **Cancer Diagnosis:** Built classifiers (Logistic Regression, Support Vector Machines, Decision Trees) on the Wisconsin Diagnostic Breast Cancer dataset, tuning regularization and optimizing F1-score.
  * **Housing Price Prediction:** Engineered regression pipelines (Multi-Layer Perceptron Neural Networks and Decision Trees) for California housing evaluation, benchmarked using RMSE.
  * **Metabolic Classification:** Multi-feature diabetes classification models exploring decision boundaries and feature importance.

---

### 5. Geometric Deep Learning & Generative Modeling
* **Geometric & Equivariant Machine Learning Suite ([`Structured-machine-learning`](projects/Structured-machine-learning/)):**  
  * **$E(3)$-Equivariant Force Fields:** Formulated molecular potential energy models (`e3nn`) preserving Euclidean symmetries with spherical harmonics and Clebsch-Gordan tensor products on MD17 Ethanol CCSD(T) trajectories.
  * **Continuous Normalizing Flows (CNF):** Built generative Conditional Flow Matching (CFM) with PaiNN equivariant vector fields for i.i.d. sampling from Boltzmann distributions, demonstrating an 88% reduction in KDE energy divergence compared to unconstrained MLPs.
  * **Topological Graph Transformers:** Benchmarked GAT, GCN, and DeepSets on MoleculeNet benchmarks (ESOL solubility and BBBP permeability) with scaffold-based stratification.

---

### 6. Data Engineering, Web Scraping & Unsupervised Learning
* **Real Estate Web Scraping & Market Analytics ([`RealEstate_Scraping_Analytics`](projects/RealEstate_Scraping_Analytics/)):**  
  Engineered automated web scraping pipelines using BeautifulSoup to extract, clean, and normalize real estate listing data. Performed exploratory data analysis through histograms, scatter correlation plots, and quartile distributions.
* **Morphological Seed Kernel Clustering ([`Wheat_Clustering`](projects/Wheat_Clustering/)):**  
  Unsupervised clustering pipeline applying K-Means and hierarchical agglomerative clustering to group seed varieties based on geometric attributes.
* **Graph Community Detection ([`CommunityDetector`](projects/CommunityDetector/)):**  
  Network analysis algorithms detecting community structures and sub-graph partitions in complex graphs.
* **Applied Linear Algebra Algorithms ([`linearAlgebra`](projects/linearAlgebra/)):**  
  Mathematical implementations of matrix factorizations, singular value decomposition (SVD), and linear system solvers.
* **Adversarial Search & Game Engines ([`Game-playing-systems`](projects/Game-playing-systems/)):**  
  Game-playing decision systems utilizing Minimax search with Alpha-Beta pruning heuristics.
* **Demographic Statistical Modeling ([`DependencyRatio`](projects/DependencyRatio/)):**  
  Statistical regression and demographic analysis tracking economic dependency metrics.

---

## Directory Structure

```text
ai-machine-learning/
├── README.md
└── projects/
    ├── AI_101/
    ├── AItools/
    ├── ANN/
    ├── CommunityDetector/
    ├── DependencyRatio/
    ├── DiabetesClassifier/
    ├── Diagnostic-Systems/
    ├── Game-playing-systems/
    ├── LifeExpectancy/
    ├── linearAlgebra/
    ├── MNIST_pytorch_classification/
    ├── RealEstate_Scraping_Analytics/
    ├── RecommenderSystem/
    ├── Semantic-Segmentation-with-Deep-Learning/
    ├── Spaceship_Titanic/
    ├── Spam-Ham/
    ├── Structured-machine-learning/
    └── Wheat_Clustering/
```

---

## Technology Stack

* **Deep Learning & Neural Networks:** PyTorch, PyTorch Geometric, `e3nn`, U-Net, Convolutional Neural Networks, Multilayer Perceptrons.
* **Machine Learning & Data Science:** Scikit-Learn, NumPy, SciPy, Pandas, Random Forest, SVM, Decision Trees, K-Means.
* **NLP & Conversational AI:** Rasa Framework, NLTK, Regular Expressions, TF-IDF.
* **Data Engineering & Visualization:** BeautifulSoup, Requests, Matplotlib, Seaborn.
