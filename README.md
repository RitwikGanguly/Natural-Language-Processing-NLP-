# Natural-Language-Processing-NLP-
> Here I will be adding all NLP theory as well as codes from very basic to Adv

- **Also Added one popular binary classification problem "Email Spam Detection"**
- **And added the classification problem "Fake News Detection"**

  ---

  # NLP Important Concept(Questions & Answer)

**Question 1: What are vocabularies, documents, and corpus in NLP?**

Document is the unit for Natural Language Processing (NLP). If an NLP project has 1000 reviews, each review is a document, so we can say that the project includes 1000 documents.

Corpus is a collection of documents. In an NLP project with 1000 reviews, the corpus consists of all 1000 reviews.

Vocabulary is a collection of unique words or tokens in a corpus. If we remove the duplicates from the 1000 reviews and there are 2000 unique words left, the 2000 unique words are the vocabulary for the NLP project.

---

**Question 2: What is tokenization?**

Tokenization is the process of breaking down the raw text into small chunks called tokens. The tokens can be characters, part of words, words, or sentences.

`word_tokenize` from nltk breaks sentences into words.

```python
# Import tokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Example sentence
text = 'Harvard University is devoted to excellence in teaching, learning, and research, and to developing leaders who make a difference globally.'

# Tokenize the sentence
tokens = word_tokenize(text)
tokens
```

We can see that the sentence text string becomes a list of words and punctuations after tokenization.

```
['Harvard',
 'University',
 'is',
 'devoted',
 'to',
 'excellence',
 'in',
 'teaching',
 ',',
 'learning',
 ',',
 'and',
 'research',
 ',',
 'and',
 'to',
 'developing',
 'leaders',
 'who',
 'make',
 'a',
 'difference',
 'globally',
 '.']
```

We can also tokenize at the sentence level using `sent_tokenize` from the nltk library.

```python
# Import sentence tokenizer
from nltk.tokenize import sent_tokenize

# A string with two sentences
texts = 'Harvard University is devoted to excellence in teaching, learning, and research, and to developing leaders who make a difference globally. Harvard is focused on creating educational opportunities for people from many lived experiences.'

# Sentence tokenization
sent_tokenize(texts)
```

The string with two sentences is broken into two individual sentences in a list after sentence tokenization.

```
['Harvard University is devoted to excellence in teaching, learning, and research, and to developing leaders who make a difference globally.',
 'Harvard is focused on creating educational opportunities for people from many lived experiences.']
```

---

**Question 3: What are stop words?**

Stop words are a collection of words that are commonly used in a language but do not provide much meaning for the text. Some examples of stop words are the, in, and at.

The popular python NLP packages such as NLTK and spaCy have a list of default stop words, but the list can be customized by adding or removing words based on project needs.

The code for listing, adding, and removing stop words from NLTK is provided.

After importing the nltk library and downloading the stopwords, the stopwords for English are saved in a variable called `stopwords`.

```python
# Import nltk
import nltk

# Download stopwords
nltk.download('stopwords')

# Stopwords list
stopwords = nltk.corpus.stopwords.words('english')

# Print the stop words
print(f'There are {len(stopwords)} default stopwords. They are {stopwords}')
```

We can see that the default English stopwords list has 179 words.

```
There are 179 default stopwords. They are ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

New stop words can be added to the list using the `extend` method.

```python
# New stop words list
newStopwords = ['newword1', 'newword2']

# Add new stop words
stopwords.extend(newStopwords)

# Print the stop words
print(f'There are {len(stopwords)} default stopwords. They are {stopwords}')
```

We can see that the number of stop words increased from 179 to 181 after adding two new stop words

.

```
There are 181 default stopwords. They are ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'newword1', 'newword2']
```

The stop words can be removed from the list using list comprehension.

```python
# Existing stop words to remove
removeStopwords = ['i', 'me', 'my', 'myself']

# Remove stop words
finalStopwords = [w for w in stopwords if w not in removeStopwords]

# Print the stop words
print(f'There are {len(finalStopwords)} default stopwords. They are {finalStopwords}')
```

The number of stop words decreased from 181 to 177 after removing four stop words, 'i', 'me', 'my', and 'myself'.

```
There are 177 default stopwords. They are ['we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

After the stop words list is finalized, we can remove the stop words from the tokenized sentence using list comprehension.

```python
# Remove Stopping Words
text_no_stopwords = [w for w in tokens if not w in finalStopwords] 
text_no_stopwords
```

Output:

```
['Harvard',
 'University',
 'devoted',
 'excellence',
 'teaching',
 ',',
 'learning',
 ',',
 'research',
 ',',
 'developing',
 'leaders',
 'make',
 'difference',
 'globally',
 '.']
```

---

**Question 4: What is N-gram?**

N-gram is n continuous sequence of tokens in a document.

The most commonly used n-grams are unigram, bigram, and trigram. Unigram has one token, bigram has two consecutive tokens, and trigram has three consecutive tokens.

When there are more than 3 consecutive tokens

, the name represents the number of tokens. For example, when there are 5 consecutive tokens, it is called 5-gram.

To implement N-gram in python, we imported `ngrams` from the nltk library.

`string` is imported to get the list of punctuation. This is because punctuation will be removed before implementing ngrams.

```python
# Import ngram
from nltk import ngrams

# Import string for the punctuation list
import string

# Remove punctuation
punctuations = list(string.punctuation)
text_no_punct = [w for w in text_no_stopwords if not w in punctuations] 
text_no_punct
```

Output:

```
['Harvard',
 'University',
 'devoted',
 'excellence',
 'teaching',
 'learning',
 'research',
 'developing',
 'leaders',
 'make',
 'difference',
 'globally']
```

After the punctuations are removed, the tokens only include word tokens. We set `n = 5` for the 5-grams.

```python
# Ngram
n = 5
nGrams = list(ngrams(text_no_punct, n))

# Print out ngram
print(nGrams)
```

The output shows that 5 consecutive tokens starting from each token are created.

```
[('Harvard', 'University', 'devoted', 'excellence', 'teaching'), ('University', 'devoted', 'excellence', 'teaching', 'learning'), ('devoted', 'excellence', 'teaching', 'learning', 'research'), ('excellence', 'teaching', 'learning', 'research', 'developing'), ('teaching', 'learning', 'research', 'developing', 'leaders'), ('learning', 'research', 'developing', 'leaders', 'make'), ('research', 'developing', 'leaders', 'make', 'difference'), ('developing', 'leaders', 'make', 'difference', 'globally')]
```

---
**Question 5: What are stemming and lemmatization?**

Both stemming and lemmatization are techniques used in Natural Language Processing (NLP) to reduce words to their base or root form. The goal is to handle different inflections of the same word and treat them as the same word, thus reducing the dimensionality of the data and simplifying text analysis.

**Stemming:**
Stemming is a process of removing suffixes or prefixes from a word to obtain its root form or stem. The resulting stem may not always be a valid word. It uses simple rules and heuristics to chop off common word endings. For example, the word "running" would be stemmed to "run," and "jumps" would become "jump."

**Lemmatization:**
Lemmatization is a more advanced technique compared to stemming. It involves reducing words to their base or dictionary form, known as the lemma. Lemmatization considers the context and part-of-speech (POS) of the word to determine the lemma accurately. For example, the word "better" would be lemmatized to "good," and "cars" would become "car."

**Example:**

Stemming:

```python
# Stemming
from nltk.stem import PorterStemmer
text_stemmed = [PorterStemmer().stem(w) for w in text_no_punct]
print(text_stemmed)
```

Output:

```
['harvard', 'univers', 'devot', 'excel', 'teach', 'learn', 'research', 'develop', 'leader', 'make', 'differ', 'global']
```

Lemmatization:

```python
# Lemmatization
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
text_lemma = [wn.lemmatize(w) for w in text_no_punct]
print(text_lemma)
```

Output:

```
['Harvard', 'University', 'devoted', 'excellence', 'teaching', 'learning', 'research', 'developing', 'leader', 'make', 'difference', 'globally']
```

---

**Question 6: What is count vectorization?**

Count vectorization is a popular technique used in NLP to convert a collection of text documents into a numerical feature matrix. It represents each document as a vector of token counts, where each token (word or n-gram) is assigned an index. The value in each cell of the matrix represents the frequency of the corresponding token in the document.

The steps involved in count vectorization are as follows:

1. Tokenization: The text is split into individual words or tokens.
2. Vocabulary Creation: A unique set of tokens is created, and each token is assigned an index.
3. Counting: The frequency of each token in the document is counted and placed in the corresponding cell of the matrix.

**Example:**

```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

# Example text
text1 = ['Data science is fun.', 'Data science helps us to make data-driven decisions.']

# Fit the vectorizer
vectorizer.fit(text1)

# Print out the vocabulary
print('Vocabulary: ')
print(vectorizer.vocabulary_)
```

Output:

```
Vocabulary: 
{'data': 0, 'science': 7, 'is': 5, 'fun': 3, 'helps': 4, 'us': 9, 'to': 8, 'make': 6, 'driven': 2, 'decisions': 1}
```

The `transform` method produces the count vector values. For example, the first column is at index 0, which represents the token "data." Because the word "data" appeared once in the first sentence and twice in the second sentence, we have the value of 1 for the first sentence and the value of 2 for the second sentence.

```python
# Get the count vector
countVector = vectorizer.transform(text1)

# Print out the count vector
print('Count vector: ')
print(countVector.toarray())
```

Output:

```
Count vector: 
[[1 0 0 1 0 1 0 1 0 0]
 [2 1 1 0 1 0 1 1 1 1]]
```

Each row of the count vector represents a document, and each column represents a token from the vocabulary. The values in the matrix indicate the frequency of each token in the corresponding document.

---

**Question 7: What is TF-IDF?**

TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical representation of words in a document based on their importance within the document and the entire corpus. The purpose of TF-IDF is to highlight words that are both frequent in a specific document and relatively rare across the entire corpus, thus helping to identify the most relevant and distinctive words in each document.

**TF (Term Frequency):** This measures the frequency of a term (word) within a document. It is calculated as the number of times a term appears in a document divided by the total number of terms in that document. The higher the TF value, the more important the term is in that specific document.

**IDF (Inverse Document Frequency):** This measures the rarity of a term across the entire corpus. It is calculated as the logarithm of the total number of documents divided by the number of documents containing the term. The higher the IDF value, the more unique and important the term is in the corpus.

The TF-IDF score is the product of TF and IDF:

**TF-IDF = TF * IDF**

The TF-IDF score provides a way to give higher weight to important words in a document while downplaying common and unimportant words.

**Example:**

```python
# TFIDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

# Example text
text1 = ['Data science is fun.', 'Data science helps us to make data-driven decisions.']

# Fit the vectorizer
tfidf.fit(text1)

# Print out the vocabulary
print('Vocabulary: ')
print(tfidf.vocabulary_)
```

Output:

```
Vocabulary: 
{'data': 0, 'science': 7, 'is': 5, 'fun': 3, 'helps': 4, 'us': 9, 'to': 8, 'make': 6, 'driven': 2, 'decisions': 1}
```

```python
# Get the TF-IDF vector
vector_tfidf = tfidf.transform(text1)

# Print out the TF-IDF vector
print('Full vector: ')
print(vector_tfidf.toarray())
```

Output:

```
Full vector: 
[[0.40993715 0.         0.         0.57615236 0.         0.57615236
  0.         0.40993715 0.         0.        ]
 [0.48719673 0.342369   0.342369   0.         0.342369   0.
  0.342369   0.24359836 0.342369   0.342369  ]]
```

In the TF-IDF vector representation, each row corresponds to a document (sentence), and each column corresponds to a unique word from the vocabulary. The values in the matrix represent the TF-IDF scores of each word in the corresponding document. Higher values indicate higher importance of the word in the document. Note that common words like "is" and "to" have lower TF-IDF scores as they are less distinctive across the entire corpus.

---

**Question 8: What is bag of words (BoW)?**

The bag of words (BoW) model is a popular technique used in NLP for text representation. It converts text data into numerical feature vectors without considering the order or structure of the words. The BoW model builds a vocabulary of unique words from the entire corpus and then counts the occurrences of each word in each document.

In the BoW model, each document is represented as a vector of word frequencies, and the position of each word in the vector corresponds to its index in the vocabulary. If a word is present in the document, its frequency (number of occurrences) is placed in the corresponding position; otherwise, the frequency is set to zero.

The BoW model is simple to implement and computationally efficient. However, it loses the sequence information of the words and does not capture the semantic relationships between them.

**Example:**

Consider the following corpus of two sentences:

```python
corpus = ['Data science is fun.', 'Data science helps us to make data-driven decisions.']
```

Using the BoW model, we first create a vocabulary from all unique words in the corpus:

```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the corpus and get the vocabulary
vectorizer.fit(corpus)
vocabulary = vectorizer.vocabulary_

print('Vocabulary: ')
print(vocabulary)
```

Output:

```
Vocabulary: 
{'data': 0, 'science': 7, 'is': 5, 'fun': 3, 'helps': 4, 'us': 9, 'to': 8, 'make': 6, 'driven': 2, 'decisions': 1}
```

Next, we use the CountVectorizer to transform the corpus into BoW vectors:

```python
# Transform the corpus into BoW vectors
bow_vectors = vectorizer.transform(corpus)

print('Bag of Words Vectors: ')
print(bow_vectors.toarray())
```

Output:

```
Bag of Words Vectors: 
[[1 0 0 1 0 1 0 1 0 0]
 [2 1 1 0 1 0 1 1 1 1]]
```

In the BoW vectors, each row corresponds to a document (sentence), and each column corresponds to a word in the vocabulary. The values in the matrix represent the frequency of each word in the corresponding document. For example, the first row represents the vector for the sentence "Data science is fun." The word "data" appears once, "science" appears once, "is" appears once, "fun" appears once, and all other words have zero frequency in this document.

---

**Question 9: What is word embedding?**

Word embedding is a representation technique used in Natural Language Processing (NLP) to map words or phrases to continuous vector spaces, where similar words are located closer to each other in the vector space. It is a way to capture the semantic meaning and relationships between words.

Word embeddings are dense and low-dimensional vector representations that are learned from large amounts of text data using techniques like Word2Vec, GloVe (Global Vectors for Word Representation), and FastText. These techniques consider the context in which words appear in sentences and learn representations that encode semantic information.

Word embeddings have several advantages over traditional methods like one-hot encoding or bag-of-words:

1. Dimensionality Reduction: Word embeddings map words into a lower-dimensional continuous space, reducing the dimensionality of the representation compared to one-hot encoding, which can be beneficial for computational efficiency.

2. Semantic Relationships: Word embeddings capture semantic relationships between words. Similar words are close to each other in the vector space, allowing for meaningful comparisons and calculations.

3. Contextual Information: Word embeddings take into account the context in which words appear. Words with similar meanings are placed close together in the vector space, making them useful for tasks like sentiment analysis and natural language understanding.

4. Handling Out-of-Vocabulary (OOV) Words: Word embeddings can handle unseen or out-of-vocabulary words

 by providing a continuous representation even for words not seen during training.

These word embeddings are used as pre-trained word embeddings in various NLP tasks, such as text classification, named entity recognition, machine translation, and sentiment analysis. They are also used as the initial input for training more complex deep learning models for NLP tasks.

---

**Word2Vec**

Word2Vec is a neural network-based technique used in natural language processing (NLP) to generate word embeddings from a training text corpus. These word embeddings represent words as dense vectors in a continuous vector space, capturing semantic relationships between words.

**Continuous Bag-of-Words (CBOW)**

CBOW is one of the learning algorithms used in Word2Vec. In the CBOW model, the objective is to predict the target word based on the surrounding context words. It takes a fixed-size window of context words and tries to predict the target word at the center of the window. CBOW is faster and performs well for frequent words but may not be as effective for rare words.

**Skip-Gram**

Skip-Gram is another learning algorithm used in Word2Vec. In this model, the target word is taken as input, and the objective is to predict the surrounding context words. Skip-Gram aims to maximize the probability of predicting context words given the target word. It is slower than CBOW but often performs better, especially for rare words, as it provides more training data for infrequent word-context pairs.

**Context (Window) Size**

The context size, also known as the window size, is a parameter in Word2Vec models. For Skip-Gram, it typically takes a target word and tries to predict context words within a window. The context size is usually set around 10 for Skip-Gram and around 5 for CBOW.

The figure below illustrates the difference between CBOW and Skip-Gram:

```
Continuous Bag-of-Words (CBOW):
Context Words -> CBOW Model -> Target Word

Skip-Gram:
Target Word -> Skip-Gram Model -> Context Words
```

**Word Vector Quality**

The quality of the learned word vectors is affected by several factors:

1. **Corpus size and quality:** The size and quality of the training dataset can significantly impact the word vector quality.

2. **Word vector size:** The dimensionality of the word vectors (embedding size) can affect the richness of semantic information captured. Larger sizes may be beneficial, but it's not always the case.

3. **Model structure and algorithm:** The chosen Word2Vec architecture (CBOW or Skip-Gram) and the training algorithm influence the final word vector quality.

The quality of the word vectors is crucial because it affects the downstream tasks’ performance. The quality of the word vectors are mainly affected by three factors:

- Corpus size and quality for the training dataset.
- The size of the word vectors. The dimensionality of the word vectors is usually more is better, but not always.
- The training model structure and algorithm.












### In this repo, there also provided the vectorization concept notebook with theory and code.

## Happy Learning 🚀
