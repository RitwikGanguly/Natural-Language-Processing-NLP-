import streamlit as st
from PIL import Image
import os

from pathlib import Path


# cur_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# prof_pic = cur_dir / "assets" / "ricky.jpg"

cur_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
prof_pic = cur_dir / "assets" / "ritwik_pic.jpg"
background = cur_dir / "resources" / "images" /"cool_background.jpg"

# Description
PAGE_TITLE = "Ritwik Ganguly"
PAGE_ICON = "📩"
NAME = "Ritwik Ganguly"
DESCRIPTION = """
I am a CSE student, currently in 4th year and enthusiastic in data science, data analysis & Machine Learning, aspire to learn new thing at every second. 
"""
EMAIL = "gangulyritwik2003@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn": "https://linkedin.com/in/ritwikganguly003",
    "GitHub": "https://github.com/RitwikGanguly",
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.pexels.com/photos/255377/pexels-photo-255377.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
background-size: cover;
background-position: top center;
}
</style>
'''


side_bgimg = '''
<style>
.stApp {
background-image: url("https://images.pexels.com/photos/2680270/pexels-photo-2680270.jpeg?auto=compress&cs=tinysrgb&w=600");
background-size: cover;
background-position: left;
}
</style>
'''
st.sidebar.markdown(side_bgimg, unsafe_allow_html=True)
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Hi!! :red[Everyone]")
st.subheader("welcome all, let's enjoy together😎😎😎😎")

# Load css, pdf and profile pic

# with open(css_file) as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

prof = Image.open(prof_pic)


# Header section

col1, col2 = st.columns(2, gap = "small")
with col1:
    st.image(prof, width=220)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.write("📩", EMAIL)

st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

st.write('\n')
st.subheader("Knowledge & Self-declaration ")
st.write(
    """
- **✔️ Currently a student, studied in 4th year**
- **✔️ Have knowledge in python 😎, still exploring and trying to grasp a good hand in python**
- **✔️ Has understanding in data science and trying to find the hardness behind the easy word ML** 
- **✔️ Has team management and problem solving capability**
- **✔️ Always try to learn new things through challenges and tasks** 
- **✔️ Always try to improve my task than the previous task which I did**
- **✔️ Find and try to make the wackiness to my strongest strength.**
"""
)
st.subheader("Project Name - Email Spam Detector")
st.subheader("Project Requirements ...............")
st.markdown(
    """
- 🍿🍿 The Things are done so far for this project......
- **🏆 All the information/data are given in a .csv file from where need to build the model**
- **🏆 Cleaning and do all the preprocessing task through pandas library of python to get a clean dataframe**
- **🏆 As the email is a text data to do the text preprocessing using Regex and build the final dataset**
- **🏆 By the NLP do the vectorization using Bag of Words(BOW) and make the vectorized matrix for the prediction**
- **🏆 Finally used Logistic Regression(LR) for the final prediction for this well-known binary classification.** 
- **🏆 Last but not the least will display a beautiful WordCloud for each user email input.**


"""
)


st.subheader("👉 THE GITHUB LINK OF THIS PROJECT:-")
st.write("Click the link 👇")
sam = '''
> [**EMAIL SPAM💀**](https://github.com/RitwikGanguly/Natural-Language-Processing-NLP-)

'''
st.markdown(sam)






