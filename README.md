# Twitter-X Sentiment Analysis

This project implements a robust, end-to-end sentiment analysis pipeline for Twitter data using classic NLP techniques and a machine learning classifier (Logistic Regression). The workflow covers everything from raw data ingestion to deployment-ready sentiment prediction, providing strong machine learning engineering and NLP best practices throughout.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Environment and Libraries](#environment-and-libraries)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Cleaning and Preprocessing](#2-data-cleaning-and-preprocessing)
  - [3. Text Preprocessing](#3-text-preprocessing)
  - [4. Handling Class Imbalance](#4-handling-class-imbalance)
  - [5. Data Visualization](#5-data-visualization)
  - [6. Feature Engineering](#6-feature-engineering)
  - [7. Model Training](#7-model-training)
  - [8. Model Evaluation](#8-model-evaluation)
  - [9. Inference: Custom Sentiment Prediction](#9-inference-custom-sentiment-prediction)
- [Results](#results)
- [How to Use](#how-to-use)
- [Key Points and Design Choices](#key-points-and-design-choices)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This sentiment analysis project classifies tweets as either positive or negative using a pipeline that includes data cleaning, text preprocessing, class balancing, feature extraction using TF-IDF (with n-grams), and Logistic Regression modeling. The notebook is designed for clarity and reproducibility and includes all visualization and evaluation steps.

---

## Dataset

- **Source:** The dataset is downloaded directly from GitHub:  
  `https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv`
- **Content:**  
  - Each row contains a tweet and its sentiment label.
  - Only two columns are used:  
    - `label` (sentiment, where 0 = positive, 1 = negative)  
    - `tweet` (tweet text)
- **Preprocessing:**  
  - Rows with missing values are dropped.
  - The columns are renamed to `sentiment` and `text` for consistency.

---

## Environment and Libraries

- **Programming Language:** Python 3
- **Main Libraries Used:**  
  - Data Handling: `pandas`, `numpy`
  - Natural Language Processing: `nltk` (for lemmatization)
  - Machine Learning: `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`

---

## Step-by-Step Workflow

### 1. Data Loading

- Loads the dataset from a remote CSV file using pandas.
- Keeps only the relevant columns: `label` and `tweet`.
- Drops any rows with missing values and resets the index.
- Renames columns for clarity.

### 2. Data Cleaning and Preprocessing

- **Stopwords:**  
  - A static set of common English stopwords is explicitly defined in the notebook to avoid runtime dependencies.
- **Lemmatization:**  
  - Lemmatizer from NLTK (`WordNetLemmatizer`) is used.
  - Required NLTK corpora are downloaded in the notebook (`wordnet`, `omw-1.4`).

### 3. Text Preprocessing

- **Cleaning Function:**  
  - All text is converted to lowercase.
  - URLs are removed using regex.
  - Non-alphabetic characters are stripped.
  - Tokenization is performed by splitting on whitespace.
  - Stopwords are removed.
  - Lemmatization is applied to each token.
- **Result:**  
  - A new column `cleaned` is added to the DataFrame, containing the cleaned and processed tweets.

### 4. Handling Class Imbalance

- **Balance Function:**  
  - The dataset is balanced so that each sentiment class (positive and negative) has equal representation.
  - This is done by upsampling the minority class to match the size of the majority class.
  - The balanced dataset is shuffled for randomness.

### 5. Data Visualization

- **Class Distribution:**  
  - Visualizes the class distribution before and after balancing using `seaborn`’s `countplot`.
  - Plots are displayed side by side for easy comparison.
  - This helps illustrate the effect of balancing on the dataset.

### 6. Feature Engineering

- **Train-Test Split:**  
  - Splits the dataset into training and testing sets (80% train, 20% test).
- **TF-IDF Vectorization:**  
  - Uses `TfidfVectorizer` from scikit-learn.
  - Considers unigrams, bigrams, and trigrams (`ngram_range=(1,3)`).
  - Limits the feature set to the top 10,000 features.

### 7. Model Training

- **Model:**  
  - Trains a Logistic Regression model with class balancing (`class_weight='balanced'`).
  - Sets the maximum iterations to 1,000 for reliable convergence.
- **Fitting:**  
  - The model is trained on the TF-IDF features generated from the training set.

### 8. Model Evaluation

- **Prediction:**  
  - The trained model predicts sentiment labels on the test set.
- **Label Conversion:**  
  - Numeric labels are converted to human-readable strings:
    - `0` → "Positive (0)"
    - `1` → "Negative (1)"
- **Metrics:**  
  - Prints classification accuracy.
  - Displays the full classification report (precision, recall, f1-score, support for each class).
- **Confusion Matrix:**  
  - Displays a confusion matrix plot for a visual breakdown of prediction performance.

### 9. Inference: Custom Sentiment Prediction

- **Custom Prediction Function:**  
  - A function (`predict_sentiment`) allows users to input any sentence or tweet and get the predicted sentiment.
  - The function applies the same cleaning and vectorization steps as during training.
- **Example Predictions:**  
  - The notebook provides sample predictions for various inputs, e.g.:
    - `"i love it"` → Positive (0)
    - `"commit criminal activities"` → Negative (1)
    - `"i hate it"` → Negative (1)
    - `"he is a very good lawyer"` → Positive (0)

---

## Results

- **Model Performance:**
  - Example classification accuracy: **96.4%**
  - Precision, recall, and f1-scores are high for both classes, indicating excellent model performance.
  - Confusion matrix confirms balanced performance after class adjustment.

- **Example Output:**
  ```
  Accuracy: 0.9644
                precision    recall  f1-score   support

    Negative (1)       0.95      0.99      0.97      5989
    Positive (0)       0.98      0.94      0.96      5899

        accuracy                           0.96     11888
       macro avg       0.97      0.96      0.96     11888
    weighted avg       0.97      0.96      0.96     11888
``` ```
## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yaser0004/Twitter-X--Sentiment-Analysis.git
cd Twitter-X--Sentiment-Analysis
```

### 2. Install Required Dependencies

Make sure you have Python 3 installed. Install the required Python packages using pip:

```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn
```

### 3. Open the Notebook

Open the Jupyter notebook in your preferred environment (e.g., Jupyter Notebook, JupyterLab, or Google Colab):

```bash
jupyter notebook Twitter(X)-Sentiment-Analysis.ipynb
```

Or upload the notebook to [Google Colab](https://colab.research.google.com/) for a cloud-based experience.

### 4. Run All Notebook Cells

- Execute each cell in order from top to bottom.
- The notebook will:
  - Download and prepare the data.
  - Clean and preprocess tweets.
  - Balance the dataset.
  - Extract features using TF-IDF.
  - Train and evaluate the Logistic Regression model.
  - Visualize the results.

### 5. Make Custom Predictions

At the end of the notebook, you’ll find a `predict_sentiment` function. Use this to classify any custom sentence or tweet.

**Example usage:**

```python
print(predict_sentiment("This product is amazing!"))         # Output: Positive (0)
print(predict_sentiment("I am very disappointed."))          # Output: Negative (1)
```

You can modify the string passed to `predict_sentiment` for your own testing.

---

**Note:**  
- All steps are contained in the notebook.  
- No manual data download or preprocessing is needed—just run the notebook and follow the instructions above.


---

## Key Points and Design Choices

- **Static Stopword List**  
  - The notebook uses an explicitly defined set of English stopwords instead of relying on external libraries or downloads.

- **Lemmatization**  
  - Lemmatization is performed using NLTK’s WordNet lemmatizer, ensuring that words are reduced to their base form.

- **Class Balancing**  
  - The balancing function ensures that the model does not become biased towards the majority class by upsampling the minority class.

- **TF-IDF with N-Grams**  
  - Feature extraction uses unigrams, bigrams, and trigrams, capturing a richer representation of tweet language and context.

- **Logistic Regression**  
  - A simple yet powerful model for text classification, with class weights balanced to handle dataset imbalance.

- **Visualization**  
  - The notebook provides clear visualizations for both class distribution and confusion matrix to aid in understanding and reporting.

- **End-to-End Pipeline**  
  - The workflow is fully contained and can be run from raw data to final predictions and deployment-ready sentiment function.

---

## Acknowledgements

- **Dataset Source:**  
  [dD2405/Twitter_Sentiment_Analysis (GitHub)](https://github.com/dD2405/Twitter_Sentiment_Analysis)

- **Tools & Libraries:**  
  - [pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/)
  - [NLTK](https://www.nltk.org/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)

---

**This project provides a robust, reproducible approach for Twitter/X sentiment analysis, and can be adapted or extended for other text classification tasks with minimal modification.**
