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

Question 7: What is TF-IDF?
TF-IDF stands for term frequency-inverse document frequency. It measures the importance of a term in a document against all the documents in a corpus. It increases proportionally to the number of times a word appears in the document and decreases by the frequency of the word in the whole corpus.

The formular is : 𝑇𝐹-𝐼𝐷𝐹(𝑡)=𝑇𝐹(𝑡)∗𝐼𝐷𝐹(𝑡)

𝑇𝐹 stands for Term Frequency, which measures how many times a term occurs in a document. We expect a term to appear more times in longer documents than in shorter ones, so the term frequency is often normalized by the total number of terms in the document. It is usually denoted by 𝑇𝐹(𝑡,𝑑), indicating that the term frequency for a term is determined by the term frequency 𝑡 and its document length 𝑑.
𝐼𝐷𝐹 stands for Inverse Document Frequency, which measures how frequently a term appears in the whole corpus. This is important because certain terms may appear a lot of times in a document but also appear a lot of times in other documents in the corpus, which makes it not so special for the document. Therefore, we need to weigh down such terms and scale up the weights of the terms not frequent across all the documents.
👉 𝐼𝐷𝐹(𝑡)=𝑙𝑜𝑔((1+𝑁)/(1+𝐷𝐹(𝑡,𝑑))+1) where 𝑁 is the total number of documents in the corpus and 𝐷𝐹(𝑡,𝑑) is the document frequency of the term.

👉 1 is added to the logarithm part of the 𝐼𝐷𝐹 to prevent zeroing out the terms that appear in every document in the corpus. When a term appears in all the documents, the numerator and denominator have the same value, so the result is 1. 𝑙𝑜𝑔1=0, so IDF equals 0 and hence 𝑇𝐹-𝐼𝐷𝐹 equals 0. Adding 1 to the equation makes the value in the logarithm always greater than 1, which prevents zeroing out terms from happening.

👉 Adding 1 to the numerator and denominator of the equation prevents zero division. We can think of it as adding an extra document with all the terms in the corpus.

# TFIDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

# Example text
text1 = ['Data science is fun.', 'Data science helps us to make data driven decisions.']

# Fit the vectorizer
tfidf.fit(text1)

# Print out the vovabulary
print('Vocabulary: ')
print(tfidf.vocabulary_)
Fitting the input text using TfidfVectorizer from sklearn automatically creates IDs for each vocabulary in the corpus.

# TFIDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

# Example text
text1 = ['Data science is fun.', 'Data science helps us to make data driven decisions.']

# Fit the vectorizer
tfidf.fit(text1)

# Print out the vovabulary
print('Vocabulary: ')
print(tfidf.vocabulary_)
Same as the count vectorization, in the example corpus with two sentences 'Data science is fun.' and 'Data science helps us to make data driven decisions.', TfidfVectorizer assigned IDs from 0 to 9 to the 10 unique words in alphabetical order.

Vocabulary: 
{'data': 0, 'science': 7, 'is': 5, 'fun': 3, 'helps': 4, 'us': 9, 'to': 8, 'make': 6, 'driven': 2, 'decisions': 1}
transform method produces the TF-IDF vector values.

# Get the TF-IDF vector
vector_tfidf = tfidf.transform(text1)

# Print out the TF-IDF vector
print('Full vector: ')
print(vector_tfidf.toarray())
Output:

Full vector: 
[[0.40993715 0.         0.         0.57615236 0.         0.57615236
  0.         0.40993715 0.         0.        ]
 [0.48719673 0.342369   0.342369   0.         0.342369   0.
  0.342369   0.24359836 0.342369   0.342369  ]]
Question 8: What is bag of words (BoW)?
The bag of words (BoW) model is a type of NLP model that represents text using the count of words. The order of the words is not considered. The predictions of the models are based on which words appear in a document and how many times they appeared.
The NLP preprocessing steps such as tokenization, removing stop words, removing punctuation, stemming or lemmatization, and vectorization are usually applied before running a bag of words (BoW) model.













### In this repo, there also provided the vectorization concept notebook with theory and code.

## Happy Learning 🚀
