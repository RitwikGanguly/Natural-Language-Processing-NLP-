import streamlit as st
from matplotlib import image
import os
import pickle as pk
import pandas as pd
import regex as re
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import random


ps = PorterStemmer()



# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "email.jpg")

model_path = os.path.join(dir_of_interest, "data", "classifier.pkl")
cv_path = os.path.join(dir_of_interest, "data", "cv.pkl")
pre_path = os.path.join(dir_of_interest, "data", "pre.pkl")
victim_path = os.path.join(dir_of_interest, "data", "victim.pkl")
font_path = os.path.join(dir_of_interest, "data", "KaushanScript-Regular.otf")



classifier = pk.load(open(model_path, "rb"))
cv = pk.load(open(cv_path, "rb"))
# preprocessing = pk.load(open(pre_path, "rb"))
victim = pk.load(open(victim_path, "rb"))


# preprocessing part
def preprocessing(raw):
    # Removing special characters and digits
    sentence = re.sub("[^a-zA-Z\s]", " ", raw)

    # change sentence to lower case
    sentence = sentence.lower()

    # tokenize into words
    tokens = sentence.split()

    # remove stop words
    clean_tokens = [t for t in tokens if t not in victim]

    # Stemming/Lemmatization

    clean_tokens = [ps.stem(word) for word in clean_tokens]

    return pd.Series([" ".join(clean_tokens), len(clean_tokens)])

st.set_page_config(page_title="RG | EmailSpam",
                   page_icon="🚀",
                   layout="wide"
)
st.markdown("<h1 style='text-align: center; color: black;'>Email Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: red;'>🦹‍♀ Check Your Mail & beware of fraud 🦹‍♀️</h2>", unsafe_allow_html=True)

page_bg_img = '''
<style>
.stApp {
background-image: url("https://www.bloggersinsights.com/images/embed_image/4mhv1.jpeg");
background-size: cover;
background-position: top center;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
img = image.imread(IMAGE_PATH)

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')

with col2:
    st.image(img, width=400, caption="Be the first to be safe 😋")

with col3:
    st.write(' ')



st.markdown("<h2 style='text-align: left; color: red;'><bold>💻 Write Your Email ‍💻</bold></h2>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left; color: darkblue;'><bold>[On the email just ctrl+A and copy-paste the whole email]</bold></h5>", unsafe_allow_html=True)

text = st.text_area("Paste the whole Email")
butt = st.button("Check ‼")


train = preprocessing(text)
sam = train[0]

test = cv.transform([sam])

res = classifier.predict(test)[0]
if butt:
    if res == "spam":
        st.subheader("SPAM EMAIL")
        st.markdown("<h5 style='text-align: left; color: red;'><bold>It's a spam email 💀💀, be alert before opening any link, or if you think you can unsuscribe the email sender.</bold></h5>", unsafe_allow_html=True)


    elif res == "ham":
        st.subheader("SAFE EMAIL(NOT SPAM)")
        st.markdown("<h5 style='text-align: left; color: red;'><bold>It's a safe email 😊, it can contain important info, read carefully. You are good to go.</bold></h5>", unsafe_allow_html=True)


    col1 = st.columns(1)
    spam_wordcloud = WordCloud(
        stopwords=victim,
        background_color='white',
        width=1600,
        height=800,
        colormap=random.choice(plt.colormaps()),
        font_path=font_path,
    ).generate(text)

    figx = plt.figure()
    figx.patch.set_facecolor('lightblue')
    plt.imshow(spam_wordcloud, interpolation='None')
    plt.title("Email Data WordCloud", fontsize=18, color="red")
    plt.axis('off')
    plt.tight_layout()


    st.pyplot(figx)
















