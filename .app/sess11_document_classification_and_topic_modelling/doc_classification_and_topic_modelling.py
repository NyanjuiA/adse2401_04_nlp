"""
=============================================================================================================
Python script to demonstrates a Document Classification and Topic Modelling
=============================================================================================================
This program demonstrates two natural language process tasks i.e. Document Classification (Supervised Learning)
 and Topic Modelling (Unsupervised Learning) using scikit-learn package.

PART 1: Document Classification (Supervised Learning)
- TF-IDF Vectorisation
- Train/Test Split
- Multinomial Naive Bayes Classification
- Accuracy Evaluation
- Classification Report
- Interaction Predictions

PART 2: Topic Modelling (Unsupervised Learning)
- TF-IDF Vectorisation
- Latent Dirichlet Allocation (LDA)
- Topic Discovery
- Topic Interpretation
- Dominant Topic Assignment



Dataset location:
    files/articles.json
    files/topics.json

Requirements:
    !pip install scikit-learn pandas numpy


Author: Nyanjui
Date: 04 June 2026
"""
# --------------------------------------------------------------------------------
# 0. Import required modules
# --------------------------------------------------------------------------------

import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------
# 1.
# --------------------------------------------------------------------------------