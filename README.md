# Introduction

This project focuses on sentiment analysis specifically tailored for movie reviews. By analyzing the sentiments expressed in these reviews, we aim to provide insights into the overall reception of movies, identify trends in audience preferences, and aid users in making informed decisions about which movies to watch.

# **Data Preprocessing**

### **Importing Libraries**

```python

import re
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

- **re**: Provides regular expression matching operations, used for text cleaning.
- **spacy**: A popular NLP library used for tokenization, lemmatization, and more. Here, it is used for lemmatization.
- **nltk.tokenize.word_tokenize**: Tokenizes text into words.
- **nltk.corpus.stopwords**: Provides a set of standard stop words in English.

### **Loading spaCy Model**

```python
nlp = spacy.load('en_core_web_sm')
```

Loads the spaCy small English model, which includes functionalities like tokenization, part-of-speech tagging, and lemmatization.

### **Defining Custom Stop Words**

```python
stop_words = set(stopwords.words('english')) - {'no', 'not'}
```

Defines a custom set of stop words by excluding 'no' and 'not' from the standard English stop words list. This ensures that negative sentiments are preserved during preprocessing.

### **Defining Contraction Mapping**

```python
contraction_mapping = {
    "ain't": "is not",
    "aren't": "are not",
    ...
    "you're": "you are",
    "you've": "you have"
}
```

Provides a dictionary for expanding contractions in the text. This helps in standardizing the text and improving the accuracy of text analysis.

### **Text Preprocessing Function**

```python
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Expand contractions
    text = ' '.join([contraction_mapping.get(word, word) for word in text.split()])

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Lemmatize tokens
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)
```

### **Step-by-Step Explanation**

1. **Convert to Lowercase**
    
    ```python
    text = text.lower()
    ```
    
    Converts all characters in the text to lowercase to ensure uniformity.
    
2. **Expand Contractions**
    
    ```python
    text = ' '.join([contraction_mapping.get(word, word) for word in text.split()])
    ```
    
    Expands contractions using the defined mapping, replacing words like "can't" with "cannot".
    
3. **Remove URLs**
    
    ```python
    pythonCopy code
    text = re.sub(r'http\S+', '', text)
    ```
    
    Removes any URLs from the text, which are usually irrelevant for sentiment analysis.
    
4. **Remove Non-Alphabetic Characters**
    
    ```python
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    ```
    
    Removes any characters that are not alphabetic or whitespace, cleaning the text further.
    
5. **Tokenize Text**
    
    ```python
    tokens = word_tokenize(text)
    ```
    
    Splits the text into individual words (tokens).
    
6. **Lemmatize Tokens**
    
    ```python
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    ```
    
    Converts each token to its base form (lemma), reducing different forms of a word to a common base.
    
7. **Remove Stopwords**
    
    ```python
    tokens = [token for token in tokens if token not in stop_words]
    ```
    
    Removes commonly used words (stopwords) that are unlikely to contribute to the sentiment of the text, while retaining 'no' and 'not'.
    
8. **Return Preprocessed Text**
    
    ```python
    return ' '.join(tokens)
    ```
    
    Joins the tokens back into a single string, providing the cleaned and standardized text.
    

## **Conclusion**

The **`preprocess_text`** function prepares raw movie review texts for sentiment analysis by converting text to lowercase, expanding contractions, removing URLs and non-alphabetic characters, tokenizing, lemmatizing, and removing stopwords. These steps ensure that the text is in a clean and consistent format for subsequent sentiment analysis.

# **Loading Data**

## **Introduction**

This section outlines the process of loading and preparing the movie review dataset for sentiment analysis. The dataset consists of movie reviews categorized as positive or negative, and the objective is to preprocess the text data and organize it along with corresponding target labels for subsequent analysis.

## **Code Explanation**

### **Setting Dataset Path**

```python
# Path to the movie review dataset
review_dataset_path = "/kaggle/input/movie-review/txt_sentoken"
```

Specifies the directory path where the movie review dataset is located.

### **Listing Directories in Dataset**

```python
print(os.listdir(review_dataset_path))
```

Lists the directories present within the dataset directory, providing an overview of the available data.

### **Setting Paths for Positive and Negative Reviews**

```python
# Path to positive and negative review folders
pos_review_folder_path = os.path.join(review_dataset_path, "pos")
neg_review_folder_path = os.path.join(review_dataset_path, "neg")
```

Constructs the paths for the directories containing positive and negative movie reviews within the dataset.

### **Listing Files in Positive and Negative Review Folders**

```python
pos_review_file_names = os.listdir(pos_review_folder_path)
neg_review_file_names = os.listdir(neg_review_folder_path)
```

Retrieves the list of filenames present in the positive and negative review directories, facilitating access to individual review files.

### **Function to Load Text from Text File**

```python
def load_text_from_textfile(path):
    with open(path, "r") as file:
        return file.read()
```

Defines a function to read and return the contents of a text file given its path.

### **Function to Load Review from Text File**

```python
def load_review_from_textfile(path):
    return load_text_from_textfile(path)
```

Wrapper function utilizing **`load_text_from_textfile`** to load and return the text content of a review file.

### **Function to Get Data and Target Labels**

```python
def get_data_target(folder_path, file_names, review_type):
    data = [preprocess_text(load_review_from_textfile(os.path.join(folder_path, file_name))) for file_name in file_names]
    target = [review_type] * len(data)
    return data, target
```

Combines the loading and preprocessing steps to retrieve preprocessed review data along with corresponding target labels. The function takes as input the folder path, list of filenames, and the type of review (positive or negative).

### **Data Preparation**

```python
# Getting preprocessed positive and negative data along with target labels
pos_data, pos_target = get_data_target(pos_review_folder_path, pos_review_file_names, "positive")
neg_data, neg_target = get_data_target(neg_review_folder_path, neg_review_file_names, "negative")

# Concatenating positive and negative data and targets
data = pos_data + neg_data
target_ = pos_target + neg_target

# Shuffling data and targets
combined = list(zip(data, target_))
random.shuffle(combined)
data[:], target_[:] = zip(*combined)
```

Prepares the dataset by obtaining preprocessed positive and negative review data along with their corresponding target labels. The positive and negative data are concatenated, and the labels are shuffled to ensure randomness in the dataset.

## **Conclusion**

This section illustrates the process of loading and preparing the movie review dataset for sentiment analysis. By organizing the data into preprocessed text and target labels, we establish the foundation for training and evaluating sentiment analysis models on the movie review data. The subsequent steps will involve feature engineering, model training, and evaluation to derive insights from the sentiment analysis of movie reviews.

# Text Vectorization

## **Introduction**

Text vectorization is a crucial step in natural language processing (NLP) tasks, including sentiment analysis. It involves converting textual data into numerical representations that machine learning algorithms can process. In this section, we employ the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to transform the preprocessed movie review text into TF-IDF features.

## **Code Explanation**

### **Importing Libraries and Splitting Data**

```python
from sklearn.model_selection import train_test_split
```

Imports the necessary library for splitting the dataset into training and testing sets.

```python
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target_, test_size=0.2, random_state=42)
```

Splits the preprocessed review data (**`data`**) and corresponding target labels (**`target_`**) into training and testing sets. The training set will be used to train the sentiment analysis model, while the testing set will be used for evaluation.

### **Initializing TF-IDF Vectorizer**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Imports the TF-IDF vectorizer from scikit-learn, which will be used to transform the text data into TF-IDF features.

```python
# Initializing TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

```

Initializes the TF-IDF vectorizer with a maximum of 5000 features. This parameter controls the maximum number of unique words (features) to consider during vectorization.

### **Transforming Text Data into TF-IDF Features**

```python
# Transforming text data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

Transforms the preprocessed text data into TF-IDF features using the initialized TF-IDF vectorizer. The **`fit_transform()`** method is used on the training set (**`X_train`**), which both learns the vocabulary and transforms the text data. The **`transform()`** method is then applied to the testing set (**`X_test`**) to transform it into TF-IDF features using the vocabulary learned from the training set.

## **Conclusion**

Text vectorization, particularly using TF-IDF, plays a crucial role in preparing textual data for sentiment analysis. By converting movie review text into numerical representations, we enable machine learning algorithms to effectively analyze and classify sentiments. The TF-IDF vectorization technique allows us to capture the importance of words in each review while considering their frequency across the entire corpus. The transformed TF-IDF features will serve as input to train and evaluate sentiment analysis models in subsequent stages of the project.

# **Model Training and Evaluation**

## **Support Vector Machine (SVM) Model**

### **Model Training**

The Support Vector Machine (SVM) model is trained using various combinations of hyperparameters:

- **C**: The regularization parameter. A smaller C encourages a larger margin separating the classes, while a larger C aims to classify all training examples correctly, potentially leading to overfitting.
- **gamma**: The gamma parameter in SVMs is **a hyperparameter that controls the shape of the decision boundary.**
    ![Untitled](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/3d156b7b-0698-4768-93c3-c41606c513e1)

    
- **kernel**: Specifies the kernel type to be used in the algorithm. The options include 'linear', 'poly', 'rbf', and 'sigmoid'. Each kernel function maps the input data into a higher-dimensional space.
    ![Untitled (1)](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/b39ae3e9-47b1-40d7-9e1f-77f1e2f71233)

   
    

### **Model Evaluation**

The SVM model is trained on the TF-IDF features of the preprocessed movie review text. For each combination of hyperparameters, the model's accuracy is evaluated on the testing set. The best accuracy achieved was with C=8, gamma=0.1, kernel='rbf', resulting in a test accuracy of 0.88. The best model is saved for future use.

![95f814c8-86c8-491a-9ce6-f33fb8b4b639](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/92fc39af-f828-42d9-a412-ce41d126325a)
![140b2687-181a-4b09-92bd-766921d6d508](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/07e7dff2-8299-46bf-ad68-a8d6fe44d968)
![782ed1fd-9952-4fb7-8a74-0d42891da37a](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/225f8490-ea62-4c7e-ae86-be52e122a1b3)
![979902a8-3246-481a-8e98-fdb858faa603](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/05f7f9d7-e75c-404e-9310-ce1bedfb2561)





## **Random Forest Model**

### **Model Training**

The Random Forest model is trained using different combinations of hyperparameters:

- **n_estimators**: The number of trees in the forest. More trees usually lead to better performance but also increase computational cost.
- **max_depth**: The maximum depth of each tree. Limiting the depth can prevent overfitting by controlling the complexity of the model.
- **min_samples_split**: The minimum number of samples required to split an internal node. Higher values can prevent the model from learning overly specific patterns.
- **min_samples_leaf**: The minimum number of samples required to be at a leaf node. This parameter prevents the creation of nodes with few samples, helping to avoid overfitting.

### **Model Evaluation**

The model's accuracy is evaluated on the testing set for each hyperparameter combination. The best accuracy achieved was with n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=4, resulting in a test accuracy of 0.87. The best model is saved for future use.
![5f51d456-149d-4724-be52-55131d0192db](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/a1e5412e-cc8a-44f7-b369-74f8f9cbe796)


![7e77a719-9519-43ef-904d-bf19b4105a3e](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/815095fe-2987-4edd-8c4b-3ef4446b3849)

## **Logistic Regression Model**

### **Model Training**

The Logistic Regression model is trained using a grid of hyperparameters:

- **C**: The inverse of regularization strength. Smaller values specify stronger regularization.
- **penalty**: The norm used in the penalization ('l1' or 'l2'). L1 regularization can lead to sparse models, while L2 tends to distribute error evenly across all terms.
- **solver**: The algorithm to use in the optimization problem. 'liblinear' is good for small datasets, while 'saga' is better for large datasets or with L1 regularization.

### **Model Evaluation**

The accuracy of the Logistic Regression model is evaluated on the testing set for each hyperparameter combination. The best accuracy achieved was with C=10, penalty='l2', solver='liblinear', resulting in a test accuracy of 0.88. The best model is saved for future use.
![a9bb9056-7877-46a0-98ec-6becacd83536](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/e13abedc-d729-44ff-b53a-c57dfd0fd50c)

![01383002-0722-4a81-b114-1b83169fd457](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/2629b500-555a-459b-9c06-5d509bf78bd9)




## **XGBoost Model**

### **Model Training**

The XGBoost model is trained using different combinations of hyperparameters:

- **n_estimators**: The number of trees to fit. More trees can improve performance but increase the risk of overfitting.
- **learning_rate**: The step size shrinkage used to prevent overfitting. Smaller values make the model more robust to overfitting but require more trees.
- **max_depth**: The maximum depth of a tree. Controls the complexity of the model; deeper trees can capture more information but might overfit.

### **Model Evaluation**

The XGBoost model's accuracy is evaluated on the testing set for each hyperparameter combination. The best accuracy achieved was with n_estimators=300, learning_rate=0.1, max_depth=5, resulting in a test accuracy of 0.84. The best model is saved for future use.
![8357386b-3cdf-41e8-a796-4e6e0041acaa](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/da7236e1-9720-493a-897a-55b6a115265d)

![7fd2d16e-b137-410f-8bd0-fbca87ad04ae](https://github.com/El-Sayed-Mustafa/sentiment_analysis_movie_review/assets/110793510/7c90d539-dd95-432d-a5a1-6df319b64bd8)


## **Visualization**

- **Heatmaps**: Heatmaps are created to visualize the accuracy scores of the models corresponding to different hyperparameter combinations. They help identify which hyperparameters lead to the best performance.
- **3D Scatter Plots**: 3D scatter plots are generated to visualize the relationship between hyperparameters and accuracy scores. Annotations are added to each point representing the accuracy score. These visualizations provide insights into how different hyperparameters affect model performance.

## **Conclusion**

This documentation outlines the training and evaluation process for four different machine learning models used for sentiment analysis of movie reviews. By experimenting with various hyperparameter combinations, the models' performance was optimized, achieving the highest accuracy rates for each:

- **SVM**: Best accuracy of 0.88 with C=8, gamma=0.1, kernel='rbf'.
- **Random Forest**: Best accuracy of 0.87 with n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=4.
- **Logistic Regression**: Best accuracy of 0.88 with C=10, penalty='l2', solver='liblinear'.
- **XGBoost**: Best accuracy of 0.84 with n_estimators=300, learning_rate=0.1, max_depth=5.

These results highlight the importance of hyperparameter tuning in improving model performance. The best models are saved for deployment and future use in sentiment analysis tasks.
