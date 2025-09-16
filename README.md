# 🎭 AI Sentiment Analysis Dashboard

A comprehensive sentiment analysis application built with advanced NLP techniques and deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Live Demo

[**Try the App Live!**](YOUR_STREAMLIT_URL_HERE) 

## ✨ Features

### 🔍 **Single Text Analysis**
- Real-time sentiment prediction
- Confidence scores with probability breakdown
- Interactive visualizations
- Text preprocessing insights

### 📋 **Batch Processing**
- Analyze multiple texts simultaneously
- Batch results with summary statistics
- Export functionality

### 📈 **Analytics Dashboard**
- Historical analysis trends
- Confidence distribution charts
- Performance metrics
- Data insights

### 🕒 **History Tracking**
- Complete analysis history
- Timestamp tracking
- Pattern recognition

## 🤖 Machine Learning Pipeline

### Data Preprocessing
- Text normalization and cleaning
- Stopword removal
- Lemmatization
- N-gram feature extraction

### Feature Engineering
- **TF-IDF Vectorization**: Advanced term frequency analysis
- **Count Vectorization**: Bag-of-words representation
- **N-grams**: Unigrams and bigrams for context

### Model Selection
- Multiple algorithms tested:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Hyperparameter optimization
- Cross-validation for robust performance

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Best Model | 85%+ | 0.85+ | 0.85+ | 0.85+ |

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **ML Libraries**: scikit-learn, NLTK
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud
- **Version Control**: Git & GitHub

## 📦 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sentiment-analysis-dashboard.git
   cd sentiment-analysis-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 🚀 Deployment on Streamlit Cloud

### Quick Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Automatic Updates**
   - Any push to your GitHub repository will automatically update your deployed app

## 📁 Project Structure

```
sentiment-analysis-dashboard/
│
├── app.py                          # Main Streamlit application
├── sentiment_analysis_model.pkl    # Trained ML model
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
├── preprocessing.py              # Data preprocessing scripts
├── model_training.py            # Model training pipeline
│
├── data/                        # Data directory
│   ├── raw/                    # Raw datasets
│   └── processed/             # Cleaned datasets
│
├── notebooks/                  # Jupyter notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── preprocessing.ipynb   # Data preprocessing
│   └── modeling.ipynb       # Model development
│
└── assets/                   # Static assets
    ├── screenshots/         # App screenshots
    └── demo/               # Demo files
```

## 🔧 Configuration

### Environment Variables
Create a `.streamlit/config.toml` file for custom configuration:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 📸 Screenshots

### Main Dashboard
![Main Dashboard](assets/screenshots/dashboard.png)

### Analysis Results
![Analysis Results](assets/screenshots/results.png)

### Batch Processing
![Batch Processing](assets/screenshots/batch.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📊 Model Performance Details

### Training Data
- **Dataset Size**: [YOUR_DATASET_SIZE]
- **Features**: Preprocessed text data
- **Labels**: Positive, Negative, Neutral

### Evaluation Results
```
Classification Report:
                precision    recall  f1-score   support

    Negative       0.85      0.84      0.85       XXX
     Neutral       0.82      0.83      0.82       XXX
    Positive       0.87      0.88      0.87       XXX

    accuracy                           0.85      XXXX
   macro avg       0.85      0.85      0.85      XXXX
weighted avg       0.85      0.85      0.85      XXXX
```

## 🐛 Known Issues

- Large text inputs (>10,000 characters) may take longer to process
- NLTK downloads required on first run

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NLTK** for natural language processing tools
- **scikit-learn** for machine learning algorithms
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations

## 📞 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

**Project Link**: [https://github.com/YOUR_USERNAME/sentiment-analysis-dashboard](https://github.com/YOUR_USERNAME/sentiment-analysis-dashboard)

**Live Demo**: [YOUR_STREAMLIT_URL_HERE](YOUR_STREAMLIT_URL_HERE)

---

⭐ **Star this repository if you found it helpful!** ⭐
