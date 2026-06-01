import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="India Sentiment Analysis", layout="wide")
st.title("🇮🇳 India Social Issues — Sentiment Analysis")
st.markdown("Analyzing **1.6 million tweets** for India-related sentiment")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("training.1600000.processed.noemoticon.csv",
                     encoding="latin-1", header=None,
                     names=["target", "ids", "date", "flag", "user", "text"])
    keywords = ["india", "indian", "delhi", "mumbai", "bangalore", "modi",
                "bollywood", "rupee", "cricket", "pollution", "farmer",
                "election", "bjp", "congress", "kashmir", "unemployment",
                "inflation", "poverty", "chennai", "kolkata", "hyderabad"]
    pattern = "|".join(keywords)
    df = df[df["text"].str.contains(pattern, case=False, na=False)].copy()
    df["sentiment"] = df["target"].map({0: "Negative", 2: "Neutral", 4: "Positive"})
    df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S PDT %Y")
    df["day"] = df["date"].dt.date

    stop_words = set(stopwords.words("english"))
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"[^a-z\s]", "", text)
        words = [w for w in text.split() if w not in stop_words and len(w) > 2]
        return " ".join(words)
    df["clean_text"] = df["text"].apply(clean_text)
    return df

df = load_data()

# Row 1 — metrics
total = len(df)
pos = len(df[df["sentiment"] == "Positive"])
neg = len(df[df["sentiment"] == "Negative"])

col1, col2, col3 = st.columns(3)
col1.metric("Total India Tweets", f"{total:,}")
col2.metric("Positive", f"{pos:,}", f"{round(pos/total*100)}%")
col3.metric("Negative", f"{neg:,}", f"{round(neg/total*100)}%")

st.divider()

# Row 2 — bar chart + word cloud
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    counts = df["sentiment"].value_counts()
    colors = [{"Positive": "#2ecc71", "Negative": "#e74c3c"}.get(x, "gray") for x in counts.index]
    fig, ax = plt.subplots(figsize=(5, 3))
    counts.plot(kind="bar", color=colors, edgecolor="none", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Tweets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Word Cloud")
    sentiment_choice = st.radio("Show words for:", ["Positive", "Negative"], horizontal=True)
    text = " ".join(df[df["sentiment"] == sentiment_choice]["clean_text"])
    cmap = "Greens" if sentiment_choice == "Positive" else "Reds"
    wc = WordCloud(width=600, height=300, background_color="white", colormap=cmap).generate(text)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()

# Row 3 — time series
st.subheader("Sentiment Over Time")
time_df = df.groupby(["day", "sentiment"]).size().unstack(fill_value=0)
fig3, ax3 = plt.subplots(figsize=(12, 4))
if "Negative" in time_df.columns:
    ax3.plot(time_df.index, time_df["Negative"], color="#e74c3c", linewidth=2, label="Negative")
    ax3.fill_between(time_df.index, time_df["Negative"], alpha=0.1, color="#e74c3c")
if "Positive" in time_df.columns:
    ax3.plot(time_df.index, time_df["Positive"], color="#2ecc71", linewidth=2, label="Positive")
    ax3.fill_between(time_df.index, time_df["Positive"], alpha=0.1, color="#2ecc71")
ax3.legend()
ax3.set_xlabel("Date")
ax3.set_ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

st.divider()

# Row 4 — search tweets
st.subheader("🔍 Search Tweets")
keyword = st.text_input("Enter a keyword (e.g. cricket, pollution, modi)")
if keyword:
    results = df[df["text"].str.contains(keyword, case=False, na=False)][["text", "sentiment"]]
    st.write(f"Found **{len(results)}** tweets")
    st.dataframe(results.head(50), use_container_width=True)