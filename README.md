# Product-Review-Analysis

The dataset taken for the analysis was taken from Amazon Reviews.csv which contained productID, userID, Title, Summary, Text etc.

# Classification of sentiment analysis
The approaches made in Product Review Analysis(sentiment analysis)can be categorized based on techniques used. In this project the Reviews are categorised based on ,
1.	Machine Learning: Dataset are to be trained beforehand Using Standard machine algorithms After which the trined model can be tested on review for categorising. 
2.	Rule based: Also known as Lexicon based Sentiment Analysis This approach tries to Extracts information from dataset and tries to asses them according to the polarity of words. There are different rules such as negation words, idioms, dictionary polarity, emoticons etc.Using Semantic orientation i.e. measurement of opinion and subjectivity of a review or comment it generates Sentiment polarity (either positive or negative).
3. Deep Learning Based: Used  CNN-Bidirectional LSTM where the CNN Model is used for feature extraction and the LSTM Model for interpreting the features across time steps. This gave us Pretty good results.

For data preprocessing, various text preprocessing techniques were used like Tokenisation , Stemming (using Porter Stemmer), Lemmatization , Stop Words Removal and POS tagging and removing unnecessary words which don't contain any sentiment. Only words which contain sentiment were kept like Verbs , Adverbs , Adjectives and Noun.

In Rule based analysis, a already trained Library from the NLTK package i.e. Sentiwordnet and Wordnet was being used which contains precomputed positive and negative sentiments of various words along with their synonyms and overall sentiment score of a review was calculated and classified.

In the Machine Learning based Analysis, Count Vectoriser was used to convert words to a vector thereby giving them some valid values according to the features they exhibit. After converting it to an np array for processing various ML algorithms were applied like Linear SVC(SVM) , Random forests , Logistic Regression and checked for accuracy.

In the Deep learning Based Analysis The Sequential model had the following structure,
Conv1D  ->  Conv1D  ->  Conv1D  ->  Max Pooling1D  ->  Bidirectional LSTM ->  Dense  ->  Dropout  ->  Dense  ->  Dropout  ->  Dense  ->  Dropout  ->  Output
Here We also used Early stoppping function to make sure the Model doesn't overfit 

