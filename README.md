🛡️ AI Fake News Detector
A high-performance machine learning web application designed to distinguish between Real and Fake news articles. This project leverages Natural Language Processing (NLP) and a Passive Aggressive Classifier trained on a massive consolidated dataset of over 92,000 articles.

📌 Project Overview
The detector combines three major public datasets to create a robust training ground. The pipeline involves rigorous text preprocessing, TF-IDF vectorization, and a real-time prediction engine exposed through a Streamlit web interface.

Key Features:
• Massive Dataset: Trained on 92,600+ labeled articles.
• Real-time Analysis: Instant verdict upon pasting news text.
• Synthetic Augmentation: Uses Ollama (Llama 3) to generate synthetic samples for better model generalization.
• Trigger Word Extraction: Identifies specific words that influence the model's decision.


🗂️ Project Structure

PROJECT_6TH_SEM/
├── .vscode/
│   └── settings.json
├── data/                       # Raw datasets (Local only)
├── models/
│   ├── fakenews_model.pkl      # Trained Passive Aggressive Classifier
│   └── tfidf_vectorizer.pkl    # Serialized TF-IDF Vectorizer
├── processed/
│   ├── master_dataset.ipynb    # EDA and initial cleaning
│   ├── new.py                  # Synthetic data generation (Ollama)
│   └── retrainbalanced.py      # Script for class balancing & retraining
└── web_app/
    └── app.py                  # Streamlit frontend & FastAPI backend-logic


📊 Datasets Used
The model's "brain" is built on a master dataset compiled from three primary sources, totaling 92,634 rows, unified into a binary format (0 = Fake, 1 = Real).

Dataset
Description
WELFake
A large-scale benchmark dataset for fake news detection.
Gen-AI Misinformation
Samples specifically targeting AI-generated fake content.
Practice Dataset
Additional labeled articles for varied linguistic patterns.

Balance Note: The final dataset is downsampled to approximately 45,000 samples per class to prevent model bias.


⚙️ How It Works
1. Data Preprocessing
Before training, the text undergoes a "cleaning" phase to remove noise:
• Standardization: All text converted to lowercase.
• Noise Removal: URLs, HTML tags, punctuation, and brackets are stripped.
• Stopword Filtering: Common English words (e.g., "the", "is") are removed using NLTK to focus on meaningful keywords.
2. Model Training
• Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features.
• Classifier: The Passive Aggressive Classifier was chosen for its efficiency with large-scale text data.
• Performance: Achieved a validation accuracy of 85.57%.
3. Web Interface
Built with Streamlit, the app takes user input, applies the saved TF-IDF transformations, and passes the vector to the .pkl model for an instant prediction.



🚀 Getting Started

Prerequisites

Ensure you have Python 3.13+ installed. It is recommended to use a Conda environment.

pip install streamlit scikit-learn pandas nltk

Running the App

cd web_app
streamlit run app.py

📈 Model Performance

The application will be available at http://localhost:8501.

Metric
Value
Accuracy
85.57%
Algorithm
Passive Aggressive Classifier
Feature Extraction
TF-IDF (max_df=0.7)
Total Samples
~92,600

🛠️ Tech Stack
• Language: Python 3.13
• ML & NLP: Scikit-learn, NLTK, Pandas
• Web UI: Streamlit
• Synthetic Data: Ollama / LLaMA3
• Environment: VS Code / Anaconda

📝 Limitations
• Context: The model analyzes linguistic patterns, not external facts or real-world evidence.
• Language: Currently optimized for English-language news only.
• Satire: High-quality satire or nuanced propaganda may occasionally bypass the classifier.

📄 License
This project is for academic and research purposes only.

