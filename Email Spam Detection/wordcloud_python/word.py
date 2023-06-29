from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random

os.chdir(r"D:\Ritwik's Download\Data Analysis")
victim = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "ff",
    "suffering",
    "I", "and"
])

text = '''

Conversation opened. 1 unread message.

Skip to content
Using Gmail with screen readers
6 of 10,617
It’s your last day to save on thousands of courses.
Inbox

Udemy <hello@students.udemy.com> Unsubscribe
8:16 AM (10 hours ago)
to me


Udemy
Last day to save
Learning that fits your lifestyle, with courses from ₹449.00.
Shop now

Sale ends in
HOURS
HOURS
MINUTES
MINUTES
SECONDS
SECONDS

Work together. Learn together.
Help upskill your organization with team access to thousands of courses.
Learn more

Check out top Data Science courses:
Machine Learning A-Z™: AI, Python & R + ChatGPT Bonus [2023]
Machine Learning A-Z™: AI, Python & R + ChatGPT Bonus [2023]
  4.51  (171,202)

Python for Data Science and Machine Learning Bootcamp
Python for Data Science and Machine Learning Bootcamp
  4.6  (132,248)

Artificial Intelligence A-Z™ 2023: Build an AI with ChatGPT4
Artificial Intelligence A-Z™ 2023: Build an AI with ChatGPT4
  4.39  (24,152)

ChatGPT - The Complete Guide to ChatGPT & OpenAI APIs
ChatGPT - The Complete Guide to ChatGPT & OpenAI APIs
  4.67  (1,413)

Complete Machine Learning & Data Science Bootcamp 2023
Complete Machine Learning & Data Science Bootcamp 2023
  4.61  (16,654)

Mastering OpenAI Python APIs: Unleash ChatGPT and GPT4
Mastering OpenAI Python APIs: Unleash ChatGPT and GPT4
  4.78  (691)

TensorFlow Developer Certificate in 2023: Zero to Mastery
TensorFlow Developer Certificate in 2023: Zero to Mastery
  4.6  (7,633)

LangChain- Develop LLM powered applications with LangChain
LangChain- Develop LLM powered applications with LangChain
  4.67  (760)

Deep Learning A-Z™ 2023: Neural Networks, AI & ChatGPT Bonus
...

[Message clipped]  View entire message
 '''

# Create and generate a word cloud image:
# finding the wordcloud
# wc = WordCloud(mask=mask, background_color='black', contour_color='white', contour_width=10,
#                colormap='Paired', width=mask.shape[1], height=mask.shape[0], random_state=42)
image_mask = np.array(Image.open('brain.jpg'))

colors = ImageColorGenerator(image_mask)

spam_wordcloud = WordCloud(
    stopwords=victim,
    background_color='white',
    width=1600,
    height=800,
    colormap=random.choice(plt.colormaps()),
    font_path="KaushanScript-Regular.otf",
).generate_from_text(text)

df_wc = spam_wordcloud.generate(text)
figx = plt.figure()
figx.patch.set_facecolor('lightblue')
plt.imshow(df_wc, interpolation='None')
plt.axis('off')
plt.tight_layout()
plt.show()
