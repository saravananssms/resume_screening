import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
import json
import os

st.set_page_config(
    page_title="Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; text-align: center; padding: 1rem 0; }
    .sub-header { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .result-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; font-size: 1.1rem; }
    .category-box { background-color: #d1ecf1; border: 2px solid #17a2b8; color: #0c5460; }
    .metric-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.2rem; border-radius: 10px; color: white; text-align: center; }
    .step-header { background: linear-gradient(90deg, #e8f5e9, #ffffff); padding: 0.8rem 1rem; border-radius: 5px; margin: 1rem 0 0.5rem 0; font-weight: 600; color: #2e7d32; border-left: 4px solid #2e7d32; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


@st.cache_resource
def load_model():
    try:
        import keras
        model = keras.models.load_model(os.path.join(MODELS_DIR, "resume_screening_model.h5"))
    except Exception:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "resume_screening_model.h5"))
        except Exception:
            return None, None, None, None

    with open(os.path.join(MODELS_DIR, "resume_tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "resume_label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    config = {}
    config_path = os.path.join(MODELS_DIR, "resume_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return model, tfidf, le, config


def clean_resume(text):
    text = str(text)
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def predict_resume(text, model, tfidf, le):
    cleaned = clean_resume(text)
    features = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(features, verbose=0)[0]
    top_indices = np.argsort(prediction)[::-1][:5]
    results = []
    for idx in top_indices:
        results.append({
            'category': le.classes_[idx],
            'confidence': float(prediction[idx])
        })
    return results


# Sidebar
st.sidebar.markdown("## Resume Screening")
page = st.sidebar.radio(
    "Navigate:",
    ["Prediction", "Implementation Details", "Model Info", "About"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- Python 3.12")
st.sidebar.markdown("- TensorFlow / Keras")
st.sidebar.markdown("- Scikit-learn")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 962 resumes")
st.sidebar.markdown("**Categories:** 25")
st.sidebar.markdown("**Model:** Dense Neural Network")
st.sidebar.markdown("**Format:** .h5 (Keras)")

# ===========================
# PREDICTION PAGE
# ===========================
if page == "Prediction":
    st.markdown('<div class="main-header">Resume Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify resumes into 25 job categories using Deep Neural Network</div>', unsafe_allow_html=True)

    model, tfidf, le, config = load_model()

    if model is None:
        st.error("Model not found! Run `python train_resume_model.py` from the project root to generate model files.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>Model</h3><p>Dense NN</p></div>', unsafe_allow_html=True)
    with col2:
        acc = config.get('accuracy', 'N/A')
        acc_str = f"{acc:.1%}" if isinstance(acc, float) else acc
        st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><p>{acc_str}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Categories</h3><p>{config.get("num_classes", 25)}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Paste Resume Text")

    resume_input = st.text_area(
        "Paste the full resume content below:",
        height=250,
        placeholder="Paste resume text here to classify into a job category..."
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        classify_btn = st.button("Classify Resume", type="primary", use_container_width=True)

    if classify_btn and resume_input.strip():
        with st.spinner("Classifying..."):
            results = predict_resume(resume_input, model, tfidf, le)

        top = results[0]
        st.markdown(
            f'<div class="result-box category-box">'
            f'<strong>Predicted Category: {top["category"]}</strong><br>'
            f'Confidence: {top["confidence"]:.1%}</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Top 5 Predictions")
        for r in results:
            col_x, col_y = st.columns([1, 3])
            with col_x:
                st.markdown(f"**{r['category']}**")
            with col_y:
                st.progress(float(r['confidence']), text=f"{r['confidence']:.1%}")

    elif classify_btn:
        st.warning("Please paste resume content to classify.")

    st.markdown("---")
    st.markdown("### Available Categories")
    if config and 'categories' in config:
        cats = config['categories']
    else:
        cats = le.classes_.tolist()
    cols = st.columns(5)
    for i, c in enumerate(cats):
        cols[i % 5].markdown(f"- {c}")


# ===========================
# IMPLEMENTATION DETAILS PAGE
# ===========================
elif page == "Implementation Details":
    st.markdown('<div class="main-header">Implementation Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Complete code walkthrough for Resume Screening</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-header">Step 1: Import Libraries & Load Data</div>', unsafe_allow_html=True)
    st.code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

resumeData = pd.read_csv('resume_dataset.csv', encoding='utf-8')
print(resumeData['Category'].unique())  # 25 categories""", language="python")

    st.markdown("**962 resumes** across **25 job categories**.")
    cat_data = {
        'Java Developer': 84, 'Testing': 70, 'DevOps Engineer': 55,
        'Python Developer': 48, 'Web Designing': 45, 'HR': 44,
        'Hadoop': 42, 'Blockchain': 40, 'Data Science': 40, 'Sales': 40
    }
    st.bar_chart(pd.DataFrame(list(cat_data.items()), columns=['Category', 'Count']).set_index('Category'))

    st.markdown('<div class="step-header">Step 2: Data Visualization</div>', unsafe_allow_html=True)
    st.code("""import seaborn as sns
plt.figure(figsize=(10,10))
ax = sns.countplot(x="Category", data=resumeData, palette="bright")
plt.title("Category vs Count")
plt.show()""", language="python")
    st.markdown("Bar plot and pie chart reveal **Java Developer** has the most resumes (84), **Advocate** the fewest (20).")

    st.markdown('<div class="step-header">Step 3: Text Cleaning</div>', unsafe_allow_html=True)
    st.code("""def clean_resume(Text):
    Text = re.sub('http\\S+\\s*', ' ', Text)        # URLs
    Text = re.sub('@\\S+', '  ', Text)              # Mentions
    Text = re.sub('[punctuation]', ' ', Text)        # Punctuation
    Text = re.sub('RT|cc', ' ', Text)               # RT/cc
    Text = re.sub('#\\S+', '', Text)                 # Hashtags
    Text = re.sub(r'[^\\x00-\\x7f]', r' ', Text)    # Non-ASCII
    Text = re.sub('\\s+', ' ', Text)                 # Extra spaces
    return Text""", language="python")

    clean_df = pd.DataFrame({
        'Pattern': ['http\\S+', '@\\S+', '[punctuation]', '#\\S+', '[^\\x00-\\x7f]', '\\s+'],
        'Removes': ['URLs', 'Mentions', 'Punctuation', 'Hashtags', 'Non-ASCII', 'Extra spaces']
    })
    st.table(clean_df)

    st.markdown('<div class="step-header">Step 4: Word Cloud & Frequency Analysis</div>', unsafe_allow_html=True)
    st.code("""from wordcloud import WordCloud
word_cloud = WordCloud(background_color="white").generate(cleaned_Sentences)
plt.imshow(word_cloud, interpolation="bilinear")""", language="python")
    top_words = pd.DataFrame({
        'Word': ['Details', 'Exprience', 'months', 'company', 'Data', 'Python', 'Skill', 'Education'],
        'Frequency': [484, 446, 376, 330, 200, 156, 166, 142]
    })
    st.bar_chart(top_words.set_index('Word'))

    st.markdown('<div class="step-header">Step 5: Label Encoding & TF-IDF</div>', unsafe_allow_html=True)
    st.code("""from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
resumeData['Category'] = le.fit_transform(resumeData['Category'])

from sklearn.feature_extraction.text import TfidfVectorizer
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
WordFeatures = word_vectorizer.fit_transform(required_Text)
# Shape: (962, 1500)""", language="python")
    st.markdown("Each resume becomes a **1500-dimensional** TF-IDF vector.")

    st.markdown('<div class="step-header">Step 6: KNN Classifier (Traditional ML)</div>', unsafe_allow_html=True)
    st.code("""clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
# Training Accuracy: 99%
# Test Accuracy:     99%""", language="python")
    st.success("**KNN Classifier: 99% Accuracy**")

    st.markdown('<div class="step-header">Step 7: Deep Learning Model</div>', unsafe_allow_html=True)
    st.code("""model = keras.Sequential([
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
model.save('resume_screening_model.h5')""", language="python")

    arch = pd.DataFrame({
        'Layer': ['Dense(512,relu)', 'BatchNorm', 'Dropout(0.4)', 'Dense(256,relu)', 'BatchNorm', 'Dropout(0.3)', 'Dense(128,relu)', 'Dropout(0.2)', 'Dense(25,softmax)'],
        'Params': ['768,512', '2,048', '0', '131,328', '1,024', '0', '32,896', '0', '3,225'],
        'Purpose': ['1st hidden layer', 'Normalize activations', '40% dropout', '2nd hidden layer', 'Normalize', '30% dropout', '3rd hidden layer', '20% dropout', 'Output: 25 categories']
    })
    st.table(arch)


# ===========================
# MODEL INFO PAGE
# ===========================
elif page == "Model Info":
    st.markdown('<div class="main-header">Model Information</div>', unsafe_allow_html=True)

    model, tfidf, le, config = load_model()

    st.markdown("### Model Architecture")
    st.code(
        "Input(1500 TF-IDF features)\n"
        "  -> Dense(512, relu) + BatchNorm + Dropout(0.4)\n"
        "  -> Dense(256, relu) + BatchNorm + Dropout(0.3)\n"
        "  -> Dense(128, relu) + Dropout(0.2)\n"
        "  -> Dense(25, softmax)", language=None
    )

    st.markdown("### Training Configuration")
    tc = pd.DataFrame({
        'Parameter': ['Optimizer', 'Loss', 'Max Epochs', 'Batch Size', 'Early Stopping', 'TF-IDF Features', 'Classes'],
        'Value': ['Adam', 'Categorical Crossentropy', '50', '32', 'patience=5', '1,500', '25']
    })
    st.table(tc)

    if config:
        st.markdown("### Training Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{config.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Epochs Trained", config.get('epochs_trained', 'N/A'))
        with col3:
            st.metric("Total Parameters", "939,033")

    st.markdown("### Comparison: Traditional ML vs Deep Learning")
    comp = pd.DataFrame({
        'Aspect': ['Model', 'Features', 'Accuracy', 'Training Time', 'File Format'],
        'KNN Classifier': ['K-Nearest Neighbors', 'TF-IDF (1500)', '99%', 'Seconds', 'sklearn pickle'],
        'Dense NN (Deep Learning)': ['4-layer Dense Network', 'TF-IDF (1500)', '~100%', 'Minutes', '.h5 (Keras)']
    })
    st.table(comp.set_index('Aspect'))


# ===========================
# ABOUT PAGE
# ===========================
elif page == "About":
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Problem Statement
    Automatically classify resumes into **25 job categories** using NLP and Deep Learning.

    ### Dataset
    - **962 resumes** across 25 categories
    - Categories include: Data Science, Java Developer, HR, Web Designing, DevOps, etc.

    ### Pipeline
    ```
    Raw Resume -> Text Cleaning -> TF-IDF (1500 features) -> Dense NN -> Category
    ```

    ### Key Results
    | Model | Accuracy |
    |-------|----------|
    | KNN (One-vs-Rest) | 99% |
    | Dense Neural Network | ~100% |

    ### How to Run
    ```bash
    # Train the model
    python train_resume_model.py

    # Launch the app
    streamlit run resume_screening-main/app.py
    ```

    ### Future Improvements
    - PDF/DOCX file upload support
    - Named Entity Recognition for skill extraction
    - BERT/Transformers for deeper understanding
    - Skill matching against job descriptions
    - Multi-language resume support
    """)
