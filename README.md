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

Please note that the code examples might need to be formatted or adjusted if they are meant to be executed as a standalone Python script. Also, ensure that the required libraries (e.g., nltk) are installed and imported correctly in your Python environment.















### In this repo, there also provided the vectorization concept notebook with theory and code.

## Happy Learning 🚀
