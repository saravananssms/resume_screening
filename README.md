# Resume Screening

A machine learning project that automatically classifies resumes into **25 job categories** using NLP and Deep Learning.

## Overview

This project implements a multi-class text classification system to automatically screen and categorize resumes. It uses two approaches:

1. **Traditional ML**: KNN (K-Nearest Neighbors) with One-vs-Rest strategy (Accuracy: 99%)
2. **Deep Learning**: Dense Neural Network with BatchNormalization saved as `.h5` model (Accuracy: ~100%)

## Dataset

| Property | Value |
|----------|-------|
| File | `resume_dataset.csv` |
| Records | 962 resumes |
| Columns | Category, Resume |
| Categories | 25 job categories |
| Encoding | UTF-8 |

### 25 Job Categories

| Category | Count | Category | Count |
|----------|-------|----------|-------|
| Java Developer | 84 | Operations Manager | 40 |
| Testing | 70 | Arts | 36 |
| DevOps Engineer | 55 | Database | 33 |
| Python Developer | 48 | Health and fitness | 30 |
| Web Designing | 45 | PMO | 30 |
| HR | 44 | Electrical Engineering | 30 |
| Hadoop | 42 | Business Analyst | 28 |
| Blockchain | 40 | DotNet Developer | 28 |
| Mechanical Engineer | 40 | Automation Testing | 26 |
| Data Science | 40 | Network Security Engineer | 25 |
| ETL Developer | 40 | Civil Engineer | 24 |
| Sales | 40 | SAP Developer | 24 |
| | | Advocate | 20 |

## Project Structure

```
resume_screening-main/
├── resumeScreening.ipynb        # Original Jupyter notebook
├── resume_dataset.csv           # Resume dataset
├── app.py                       # Streamlit GUI application
├── README.md                    # This file
└── Resume_Screening_Seminar_Document.docx  # Seminar documentation
```

## Pipeline

```
Raw Resumes → Text Cleaning → TF-IDF Vectorization (1500 features) → Model Training → Category Prediction
```

### 1. Text Cleaning
- Remove URLs, @mentions, punctuation, hashtags
- Remove non-ASCII characters and extra whitespace

### 2. Feature Extraction
```python
from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500
)
```

### 3. Traditional ML Approach (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
```
**Result: Accuracy = 99%**

### 4. Deep Learning Approach (Keras .h5)
```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(1500,)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(25, activation='softmax')
])
```

## Model Architecture

| Layer | Type | Output Shape | Parameters | Purpose |
|-------|------|-------------|------------|---------|
| 1 | Dense(512, relu) | (None, 512) | 768,512 | First hidden layer |
| 2 | BatchNormalization | (None, 512) | 2,048 | Normalize activations |
| 3 | Dropout(0.4) | (None, 512) | 0 | Prevent overfitting |
| 4 | Dense(256, relu) | (None, 256) | 131,328 | Second hidden layer |
| 5 | BatchNormalization | (None, 256) | 1,024 | Normalize activations |
| 6 | Dropout(0.3) | (None, 256) | 0 | Regularization |
| 7 | Dense(128, relu) | (None, 128) | 32,896 | Third hidden layer |
| 8 | Dropout(0.2) | (None, 128) | 0 | Light regularization |
| 9 | Dense(25, softmax) | (None, 25) | 3,225 | Output: 25 categories |

**Total Parameters: 939,033 (3.58 MB)**

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn keras tensorflow nltk wordcloud seaborn matplotlib
```

### Train the Model
```bash
python train_resume_model.py
```
This generates:
- `models/resume_screening_model.h5` - Keras model
- `models/resume_tfidf.pkl` - TF-IDF vectorizer
- `models/resume_label_encoder.pkl` - Label encoder
- `models/resume_config.json` - Model configuration

### Run the GUI Application
```bash
streamlit run resume_screening-main/app.py
```
Open `http://localhost:8501` in your browser.

## Technologies Used

- **Python 3.12**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - ML models, TF-IDF, metrics
- **TensorFlow/Keras** - Deep learning model
- **NLTK** - NLP preprocessing, tokenization
- **WordCloud** - Word frequency visualization
- **Seaborn & Matplotlib** - Data visualization
- **Streamlit** - Web GUI application

## Results

| Model | Metric | Score |
|-------|--------|-------|
| KNN (One-vs-Rest) | Accuracy | 99% |
| Dense Neural Network | Accuracy | ~100% |

## Exploratory Data Analysis

- **Bar Plot**: Category vs Resume Count distribution
- **Pie Chart**: Proportional category representation
- **Word Cloud**: Most frequent terms across resumes
- **Frequency Analysis**: Top 50 most common words

## Future Improvements

- Add PDF/DOCX file upload for direct resume analysis
- Implement Named Entity Recognition for skill extraction
- Build skill-matching against job descriptions
- Use BERT/Transformers for deeper semantic understanding
- Add multi-language resume support
- Integrate with Applicant Tracking Systems (ATS)
