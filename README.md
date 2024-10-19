# 🎯 Product Review Analysis: Sentiment Classification

This project focuses on **sentiment analysis** of product reviews using a variety of approaches, including **Machine Learning**, **Rule-based** methods, and **Deep Learning** techniques. It processes customer reviews to categorize them as either positive or negative, helping businesses and researchers better understand consumer opinions.
<center> <img src = "https://github.com/sandipanrakshit34/SentimentSphere---Intelligent-Product-Review-Classifier/blob/main/artificial-intelligence-in-product.png" width = 100%>
## 🚀 Project Overview

In this project, three different approaches are used to classify the sentiment of product reviews:

1. **Machine Learning Based Sentiment Analysis**: 
   - Trained the dataset using standard machine learning algorithms.
   - Models like **Linear SVC (SVM)**, **Random Forests**, and **Logistic Regression** were applied.
   - Used **Count Vectorizer** to convert textual data into numerical vectors for model processing.

2. **Rule-Based (Lexicon-Based) Sentiment Analysis**: 
   - Leveraged pre-defined rules and sentiment lexicons such as **SentiWordNet** and **WordNet** from the **NLTK** library.
   - This method calculates sentiment polarity by analyzing individual words and their associated sentiment scores.
   - Features include negation handling, idioms, emoticons, and dictionary polarity.

3. **Deep Learning Based Sentiment Analysis**: 
   - Applied a hybrid model using **CNN** for feature extraction and **Bidirectional LSTM** for capturing the temporal dependencies of features.
   - The deep learning model had the following architecture: 
     ```
     Conv1D -> Conv1D -> Conv1D -> MaxPooling1D -> Bidirectional LSTM -> Dense -> Dropout -> Dense -> Dropout -> Output
     ```
   - Used **early stopping** to prevent overfitting.

## 🔧 Data Preprocessing Techniques

Text preprocessing is a crucial step in preparing the data for sentiment analysis. Several preprocessing techniques were applied:

- **Tokenization**: Breaking down the text into individual words (tokens).
- **Stemming**: Used **Porter Stemmer** to reduce words to their root form.
- **Lemmatization**: Extracting the base form of words using **WordNet Lemmatizer**.
- **Stop Words Removal**: Removed common words that do not contribute to sentiment (e.g., "the", "is").
- **POS Tagging**: Retained only words with significant sentiment like verbs, adjectives, adverbs, and nouns, and removed unnecessary words.

## 💡 Approach Breakdown

### 1. **Machine Learning-Based Analysis**
   - Utilized **Count Vectorizer** to transform the text data into a numerical format.
   - Applied various machine learning algorithms:
     - **Linear SVC (Support Vector Classifier)**
     - **Random Forests**
     - **Logistic Regression**
   - The performance of each model was evaluated and compared for accuracy.

### 2. **Rule-Based (Lexicon-Based) Analysis**
   - Utilized the **SentiWordNet** and **WordNet** lexicons.
   - This approach calculates the overall sentiment score by considering the sentiment polarity of each word in a review.
   - Handled special cases like negation, idioms, and emoticons.
   - Generated the sentiment polarity as **positive** or **negative** based on the combined score.

### 3. **Deep Learning-Based Analysis**
   - Built a **CNN-Bidirectional LSTM** architecture for sentiment classification.
   - CNN layers were responsible for extracting features from the input text.
   - **Bidirectional LSTM** captured temporal dependencies and context in the reviews.
   - Model architecture:
     - **Conv1D -> MaxPooling1D -> Bidirectional LSTM -> Dense -> Dropout -> Dense -> Dropout -> Output**
   - Used **Early Stopping** to prevent overfitting and improve the model’s generalization.

## 📈 Results & Evaluation

Each of the approaches was evaluated for performance, and the accuracy and F1-scores were compared to determine the best technique for product review sentiment analysis.

- **Machine Learning Models**: Achieved significant accuracy with Linear SVC and Logistic Regression models.
- **Rule-Based**: Provided fast and interpretable results using pre-defined rules and lexicons.
- **Deep Learning Model**: Gave impressive results by learning complex patterns in the text, with the CNN-Bidirectional LSTM outperforming the traditional ML models in terms of accuracy.

## ⚙️ Technologies and Libraries Used

- **Programming Language**: Python
- **Libraries**:
  - **NLTK** for rule-based sentiment analysis using SentiWordNet and WordNet.
  - **Scikit-Learn** for machine learning models and Count Vectorizer.
  - **TensorFlow** / **Keras** for building and training deep learning models.
  - **Pandas** and **NumPy** for data handling and manipulation.
  
## Author

- [@sandipanrakshit34](https://github.com/sandipanrakshit34)

##


## 🛠️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sandipanrakshit34/product-review-analysis.git
   cd product-review-analysis
