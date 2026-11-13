# SMS Spam Detection Using Supervised Machine Learning

## Project Overview

This project focuses on building a Machine Learning classification model to identify whether an SMS message is **Spam** or **Ham (Not Spam)**.

The model learns patterns from labelled SMS messages using Supervised Learning and predicts the category of new, unseen messages.

The project uses **TF-IDF Vectorization** to convert text messages into numerical features and the **Multinomial Naive Bayes** algorithm for classification.

## Objectives

- Understand and explore a real-world SMS dataset.
- Perform data cleaning and preprocessing.
- Convert text messages into numerical features using TF-IDF.
- Train a supervised Machine Learning classification model.
- Predict whether new SMS messages are Spam or Ham.
- Evaluate model performance using classification metrics.

## Dataset

**Dataset:** SMS Spam Collection

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains labelled SMS messages belonging to two categories:

- `ham` – Legitimate messages.
- `spam` – Unwanted or promotional messages.

After removing duplicate records, the dataset used in this project contains **5,169 SMS messages**.

## Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning implementation |
| TF-IDF Vectorizer | Text feature extraction |
| Multinomial Naive Bayes | Classification algorithm |
| Matplotlib | Data visualization |

## Project Workflow

### 1. Data Collection
Downloaded the SMS Spam Collection dataset from the UCI Machine Learning Repository.

### 2. Data Exploration
- Checked dataset dimensions using `df.shape`.
- Inspected column information using `df.info()`.
- Analysed Spam and Ham distribution using `value_counts()`.

### 3. Data Cleaning
- Checked for missing values.
- Identified and removed duplicate records.
- Converted class labels into numerical values:
  - Ham = 0
  - Spam = 1

### 4. Text Preprocessing
- Converted messages into lowercase.
- Removed punctuation and special characters.
- Prepared text for feature extraction.

### 5. Feature Extraction
Applied **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert SMS messages into numerical feature vectors that the Machine Learning model can process.

### 6. Train-Test Split
Split the dataset into:
- 80% Training Data
- 20% Testing Data

Used stratified sampling to preserve the approximate distribution of Spam and Ham messages.

### 7. Model Training
Implemented the **Multinomial Naive Bayes** algorithm.

Trained the model using the TF-IDF training features and their corresponding labels.

### 8. Prediction
Tested the trained model using new SMS messages to predict whether they were Spam or Ham.

### 9. Model Evaluation
Evaluated the model using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Model Performance

| Evaluation Metric | Result |
|---|---:|
| Model | Multinomial Naive Bayes |
| Feature Extraction | TF-IDF |
| Train-Test Split | 80:20 |
| Accuracy | **94.97%** |

*Accuracy is based on the test split used in this experiment. Other evaluation metrics should be reviewed alongside accuracy to assess classification performance.*

## Sample Predictions

| SMS Message | Predicted Category |
|---|---|
| Congratulations! You won a free cash prize | Spam |
| Hey, are we meeting tomorrow? | Ham |
| Claim your lottery reward now | Spam |

## Key Learnings

Through this project, I gained practical experience in:

- Understanding Supervised Machine Learning.
- Working with real-world labelled datasets.
- Data exploration and preprocessing.
- Text cleaning and feature extraction.
- Understanding TF-IDF vectorization.
- Implementing the Naive Bayes classification algorithm.
- Training and testing Machine Learning models.
- Evaluating classification performance.
- Making predictions on new text messages.

## Conclusion

This project demonstrates the application of Supervised Machine Learning for text classification using Python.

By combining TF-IDF feature extraction with the Multinomial Naive Bayes algorithm, the model can classify SMS messages into Spam and Ham categories.

This project provided hands-on experience with the end-to-end Machine Learning workflow, from data preparation to model evaluation.

---

**Project Category:** Supervised Machine Learning  
**Domain:** Natural Language Processing (NLP) / Text Classification  
**Algorithm:** Multinomial Naive Bayes
