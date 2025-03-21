import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import base64
from io import BytesIO

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="wide"
)

# App title and description
st.title("📱 SMS Spam Detection")
st.markdown("""
This application uses machine learning to classify SMS messages as spam or legitimate (ham).
Upload your dataset or use the default UCI SMS Spam Collection dataset to train the model.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Dataset Exploration", "Model Training", "Spam Detection"]
selection = st.sidebar.radio("Go to", pages)

# Function to preprocess text
def preprocess_text(text):
    # Handle non-string inputs
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Function to get dataset
@st.cache_data
def get_data():
    # Default dataset - UCI SMS Spam Collection
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    try:
        df = pd.read_csv(url, sep='\t', names=['label', 'message'], encoding='latin-1')
        return df
    except Exception as e:
        st.error(f"Failed to load the default dataset: {str(e)}")
        st.info("Please upload your own dataset.")
        return None

# Function to create download link for trained model
def get_model_download_link(model, filename="spam_model.pkl"):
    """Generates a link to download the trained model"""
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/pickle;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href

# Function to try loading a CSV file with different encodings
def try_read_csv(file, encoding=None):
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    if encoding:
        try:
            return pd.read_csv(file, encoding=encoding), encoding
        except Exception as e:
            st.error(f"Error loading file with {encoding} encoding: {str(e)}")
            return None, None
    
    # Try each encoding
    for enc in encodings_to_try:
        try:
            return pd.read_csv(file, encoding=enc), enc
        except Exception:
            continue
    
    # If all encodings fail
    st.error("Failed to load the file with any of the common encodings.")
    return None, None

# Dataset Exploration Page
if selection == "Dataset Exploration":
    st.header("Dataset Exploration")
    
    # Option to upload custom dataset
    uploaded_file = st.file_uploader("Upload your own dataset (CSV format with 'label' and 'message' columns)", type="csv")
    
    df = None
    if uploaded_file is not None:
        encoding_option = st.selectbox(
            "Select file encoding",
            options=["auto-detect", "utf-8", "latin-1", "cp1252", "ISO-8859-1"],
            index=0
        )
        
        if encoding_option == "auto-detect":
            df, detected_encoding = try_read_csv(uploaded_file)
            if df is not None:
                st.success(f"Custom dataset loaded successfully with auto-detected {detected_encoding} encoding!")
        else:
            df, _ = try_read_csv(uploaded_file, encoding_option)
            if df is not None:
                st.success(f"Custom dataset loaded successfully with {encoding_option} encoding!")
    else:
        df = get_data()
        if df is not None:
            st.success("Default UCI SMS Spam Collection dataset loaded!")
    
    if df is not None:
        # Check if required columns exist
        required_cols = ['label', 'message']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Dataset is missing required columns: {', '.join(missing_cols)}")
            st.info("Please ensure your dataset has 'label' and 'message' columns.")
        else:
            # Display dataset info
            st.subheader("Dataset Overview")
            st.write(f"Dataset Shape: {df.shape}")
            st.write(f"Number of Spam messages: {len(df[df['label']=='spam'])}")
            st.write(f"Number of Ham messages: {len(df[df['label']=='ham'])}")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Distribution of spam vs ham
            st.subheader("Distribution of Spam vs Ham")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='label', data=df, ax=ax)
            plt.title('Distribution of Spam vs Ham Messages')
            st.pyplot(fig)
            
            # Message length analysis
            st.subheader("Message Length Analysis")
            df['length'] = df['message'].apply(len)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='length', hue='label', bins=50, kde=True, ax=ax)
            plt.title('Distribution of Message Lengths')
            plt.xlabel('Message Length')
            st.pyplot(fig)
            
            # Word cloud (if wordcloud package is available)
            try:
                from wordcloud import WordCloud
                
                st.subheader("Word Clouds")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Spam Messages")
                    spam_words = ' '.join(df[df['label'] == 'spam']['message'].astype(str))
                    spam_wordcloud = WordCloud(width=800, height=500, 
                                              background_color='white', 
                                              max_words=100,
                                              contour_width=3).generate(spam_words)
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(spam_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                with col2:
                    st.write("Ham Messages")
                    ham_words = ' '.join(df[df['label'] == 'ham']['message'].astype(str))
                    ham_wordcloud = WordCloud(width=800, height=500, 
                                             background_color='white', 
                                             max_words=100,
                                             contour_width=3).generate(ham_words)
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(ham_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            except ImportError:
                st.info("WordCloud package not available. Install it to see word clouds.")

# Model Training Page
elif selection == "Model Training":
    st.header("Model Training")
    
    # Get data
    uploaded_file = st.file_uploader("Upload your own dataset (CSV format with 'label' and 'message' columns)", type="csv")
    
    df = None
    if uploaded_file is not None:
        encoding_option = st.selectbox(
            "Select file encoding",
            options=["auto-detect", "utf-8", "latin-1", "cp1252", "ISO-8859-1"],
            index=0
        )
        
        if encoding_option == "auto-detect":
            df, detected_encoding = try_read_csv(uploaded_file)
            if df is not None:
                st.success(f"Custom dataset loaded successfully with auto-detected {detected_encoding} encoding!")
        else:
            df, _ = try_read_csv(uploaded_file, encoding_option)
            if df is not None:
                st.success(f"Custom dataset loaded successfully with {encoding_option} encoding!")
    else:
        df = get_data()
        if df is not None:
            st.success("Default UCI SMS Spam Collection dataset loaded!")
    
    if df is not None:
        # Check if required columns exist
        required_cols = ['label', 'message']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Dataset is missing required columns: {', '.join(missing_cols)}")
            st.info("Please ensure your dataset has 'label' and 'message' columns.")
        else:
            # Data preprocessing
            st.subheader("Data Preprocessing")
            
            with st.spinner("Preprocessing data..."):
                # Convert labels to binary (0 for ham, 1 for spam)
                df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
                
                # Apply preprocessing to messages
                st.text("Applying text preprocessing...")
                df['processed_message'] = df['message'].apply(preprocess_text)
                
                st.write("Sample of preprocessed data:")
                st.dataframe(df[['message', 'processed_message', 'label']].head())
            
            # Feature extraction and model training
            st.subheader("Model Training")
            
            # Model selection
            model_option = st.selectbox(
                "Select a classification model",
                ["Multinomial Naive Bayes", "Logistic Regression", "Random Forest"]
            )
            
            # Train-test split ratio
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            
            # Feature extraction options
            max_features = st.slider("Maximum number of features for TF-IDF", 1000, 10000, 5000)
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        df['processed_message'], 
                        df['label_binary'],
                        test_size=test_size,
                        random_state=42
                    )
                    
                    # Feature extraction
                    vectorizer = TfidfVectorizer(max_features=max_features)
                    X_train_tfidf = vectorizer.fit_transform(X_train)
                    X_test_tfidf = vectorizer.transform(X_test)
                    
                    # Model selection and training
                    if model_option == "Multinomial Naive Bayes":
                        from sklearn.naive_bayes import MultinomialNB
                        model = MultinomialNB()
                    elif model_option == "Logistic Regression":
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(max_iter=1000)
                    elif model_option == "Random Forest":
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=100)
                    
                    # Train model
                    model.fit(X_train_tfidf, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_tfidf)
                    
                    # Evaluate model
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Display results
                    st.subheader("Model Evaluation")
                    st.write(f"Model Accuracy: {accuracy:.4f}")
                    
                    st.write("Classification Report:")
                    st.text(report)
                    
                    st.write("Confusion Matrix:")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Ham', 'Spam'], 
                               yticklabels=['Ham', 'Spam'])
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig)
                    
                    # Save model and vectorizer
                    st.session_state['model'] = model
                    st.session_state['vectorizer'] = vectorizer
                    
                    # Provide download link for the model
                    st.markdown(get_model_download_link(model), unsafe_allow_html=True)
                    
                    st.success("Model training completed! You can now go to the Spam Detection page to test the model.")

# Spam Detection Page
else:  # Spam Detection
    st.header("Spam Detection")
    
    if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
        st.warning("Please train a model first on the Model Training page.")
    else:
        st.success("Model loaded successfully!")
        
        # Text input for prediction
        user_input = st.text_area("Enter an SMS message to classify:", "")
        
        if st.button("Classify Message"):
            if user_input:
                # Preprocess input
                processed_input = preprocess_text(user_input)
                
                # Transform input
                input_tfidf = st.session_state['vectorizer'].transform([processed_input])
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_tfidf)[0]
                prediction_proba = st.session_state['model'].predict_proba(input_tfidf)[0]
                
                # Display result
                st.subheader("Classification Result:")
                
                if prediction == 1:
                    st.error(f"📵 This message is classified as SPAM with {prediction_proba[1]:.2%} confidence.")
                else:
                    st.success(f"✅ This message is classified as HAM (legitimate) with {prediction_proba[0]:.2%} confidence.")
                
                # Explanation
                st.subheader("Message Analysis:")
                
                # Show top contributing words
                if hasattr(st.session_state['vectorizer'], 'get_feature_names_out'):
                    feature_names = st.session_state['vectorizer'].get_feature_names_out()
                else:
                    feature_names = st.session_state['vectorizer'].get_feature_names()
                
                # Get the words in the message that are in the vocabulary
                message_words = processed_input.split()
                vocab_words = [word for word in message_words if word in feature_names]
                
                if vocab_words:
                    st.write("Top words contributing to classification:")
                    word_indices = [list(feature_names).index(word) for word in vocab_words if word in feature_names]
                    
                    if hasattr(st.session_state['model'], 'feature_log_prob_'):
                        # For Naive Bayes
                        word_importance = [(word, st.session_state['model'].feature_log_prob_[1, idx] - 
                                          st.session_state['model'].feature_log_prob_[0, idx]) 
                                         for word, idx in zip(vocab_words, word_indices)]
                        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        # Display top words
                        for word, importance in word_importance[:10]:
                            if importance > 0:
                                st.write(f"- '{word}': More likely to appear in spam messages")
                            else:
                                st.write(f"- '{word}': More likely to appear in ham messages")
                    else:
                        st.write("Word importance analysis not available for this model type.")
                else:
                    st.write("No significant words found in the vocabulary.")
            else:
                st.warning("Please enter a message to classify.")
        
        # Batch classification
        st.subheader("Batch Classification")
        st.write("Upload a CSV file with messages to classify in bulk.")
        
        batch_file = st.file_uploader("Upload CSV file (should have a 'message' column)", type="csv", key="batch_file")
        
        if batch_file is not None:
            encoding_option = st.selectbox(
                "Select file encoding",
                options=["auto-detect", "utf-8", "latin-1", "cp1252", "ISO-8859-1"],
                index=0,
                key="batch_encoding"
            )
            
            batch_df = None
            if encoding_option == "auto-detect":
                batch_df, detected_encoding = try_read_csv(batch_file)
                if batch_df is not None:
                    st.success(f"Batch file loaded successfully with auto-detected {detected_encoding} encoding!")
            else:
                batch_df, _ = try_read_csv(batch_file, encoding_option)
                if batch_df is not None:
                    st.success(f"Batch file loaded successfully with {encoding_option} encoding!")
            
            if batch_df is not None:
                if 'message' not in batch_df.columns:
                    st.error("CSV file must contain a 'message' column.")
                else:
                    if st.button("Classify All Messages"):
                        with st.spinner("Classifying messages..."):
                            # Preprocess messages
                            batch_df['processed_message'] = batch_df['message'].apply(preprocess_text)
                            
                            # Transform and predict
                            batch_tfidf = st.session_state['vectorizer'].transform(batch_df['processed_message'])
                            batch_df['prediction'] = st.session_state['model'].predict(batch_tfidf)
                            batch_df['prediction_label'] = batch_df['prediction'].map({0: 'HAM', 1: 'SPAM'})
                            
                            # Calculate probabilities
                            proba = st.session_state['model'].predict_proba(batch_tfidf)
                            batch_df['confidence'] = [max(p) for p in proba]
                            
                            # Display results
                            st.write("Classification Results:")
                            st.dataframe(batch_df[['message', 'prediction_label', 'confidence']])
                            
                            # Summary
                            st.write(f"Total messages: {len(batch_df)}")
                            st.write(f"Spam messages: {len(batch_df[batch_df['prediction'] == 1])}")
                            st.write(f"Ham messages: {len(batch_df[batch_df['prediction'] == 0])}")
                            
                            # Download results
                            csv = batch_df[['message', 'prediction_label', 'confidence']].to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="classification_results.csv">Download Results as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("SMS Spam Detection App | Built with Streamlit and Scikit-learn")