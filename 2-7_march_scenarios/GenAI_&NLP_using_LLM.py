
# In[1]:


# # 📘 Scenario: Corporate Expansion Analysis
# # Context:
# # You are part of a business intelligence team at a multinational company. Your manager has asked you to analyze press
# releases and news articles to identify key entities such as organizations, people, locations, dates, and numbers. This
#  information will help the company track competitors’ moves and plan strategy.
# # Task:
# # You receive the following press release excerpt:
# # "Apple Inc. plans a new office in Hyderabad, India. Tim Cook announced this in March 2023. The site will create 5,000 jobs
# and focus on innovative technologies in supply chain analytics."

# # Your job is to run Named Entity Recognition (NER) using spaCy to automatically extract structured insights.


import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

text = """
Apple Inc. plans a new office in Hyderabad, India. Tim Cook announced this in March 2023.
The site will create 5,000 jobs and focus on innovative technologies in supply chain analytics.
"""

# Load the spaCy NLP pipeline (make sure the model is installed)
# Example: python -m spacy download en_core_web_sm
import spacy

nlp = spacy.load("en_core_web_sm")

# Example text for entity extraction
text = """
Apple Inc. was founded by Steve Jobs and Steve Wozniak in California.
It is headquartered in Cupertino and has offices in New York and London.
In 2023, Apple reported revenue of $394 billion.
"""

# Process the text with spaCy NLP pipeline
# This performs tokenization, POS tagging, dependency parsing, and NER
doc = nlp(text)

print("Named Entities (text, label):")

for ent in doc.ents:
    # ent.text  -> the detected entity
    # ent.label_ -> the category of entity (ORG, PERSON, GPE etc.)
    print(f"{ent.text:25} -> {ent.label_}")

from collections import defaultdict

# defaultdict automatically creates empty lists for new keys
by_label = defaultdict(list)

# Loop through entities and group them by label
for ent in doc.ents:
    by_label[ent.label_].append(ent.text)

# -------------------------------------------------------------
# Print grouped entities
# sorted(set(items)) removes duplicates and sorts results
# -------------------------------------------------------------
print("\nEntities grouped by label:")

for label, items in by_label.items():
    print(f"{label}: {sorted(set(items))}")

# In[2]:
# Scenario: Sports Event Analysis
# Context:
# You are part of a sports analytics team working for a global media company. Your manager has asked you to analyze match reports and press releases to identify key entities such as teams, players, locations, dates, and numbers. This information will help the company build automated summaries and highlight reels for fans.
# Task:
# You receive the following match report excerpt:
# "Manchester United defeated Real Madrid 3-2 at Wembley Stadium, London. Cristiano Ronaldo scored twice, while Marcus Rashford netted the winning goal in July 2024"
# Your job is to run Named Entity Recognition (NER) using spacy to automatically extract structured insights.
# 14:44
# Type a message

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Match report text
text = """Manchester United defeated Real Madrid 3-2 at Wembley Stadium, London.
Cristiano Ronaldo scored twice, while Marcus Rashford netted the winning goal in July 2024"""

# Process text
doc = nlp(text)

# Extract entities
print("Named Entities Found:\n")

for ent in doc.ents:
    print(ent.text, "->", ent.label_)


# In[3]:
# 📘 Scenario: Smart Manufacturing Insights
# Context:
# You are part of a data science team in a smart manufacturing company. Your manager wants you to explore how different
# technical concepts (like maintenance, telemetry, quality control, supply chain) are related in company documents.
# By training a small Word2Vec model, you can uncover semantic similarities between words and measure how closely they are
#  connected. This helps in building recommendation systems, knowledge graphs, or even automated reporting tools.
# Task:
# Given a short corpus describing manufacturing operations, train a Word2Vec model and analyze word similarities.


# Word embeddings with gensim Word2Vec
# Setup: python -m pip install gensim nltk
!pip install gensim


import nltk
nltk.download("punkt")
nltk.download("punkt_tab") # Added to resolve LookupError
from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models import Word2Vec

corpus = """
Manufacturing relies on predictive maintenance and supply chain optimization.
Data engineers build pipelines, while analysts monitor KPIs and anomalies.
Robotics and IoT sensors stream telemetry to cloud databases for real-time insights.
Quality control uses computer vision to detect defects on the shop floor.
"""
sentences = [word_tokenize(s.lower()) for s in sent_tokenize(corpus)]

model = Word2Vec(
    sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=2,
    sg=1
)

# Explore similar words
for target in ["maintenance", "supply", "quality", "telemetry"]:
    print(f"\nTop similar to '{target}':")
    try:
        for w, score in model.wv.most_similar(target, topn=5):
            print(f"{w:15} -> {score:.3f}")
    except KeyError:
        print("Word not in vocabulary.")

# Cosine similarity between pairs
from numpy import dot
from numpy.linalg import norm

def cosine(u, v):
    return dot(u, v) / (norm(u) * norm(v))

pairs = [("maintenance", "telemetry"),
         ("quality", "defects"),
         ("supply", "optimization")]

print("\nCosine similarities:")
for a, b in pairs:
    try:
        sim = cosine(model.wv[a], model.wv[b])
        print(f"{a:12} ~ {b:12} -> {sim:.3f}")
    except KeyError:
        print(f"Missing word: {a} or {b}")

# In[4]:
# ==========================================
# UNIVERSITY AI HELPDESK CHATBOT
# Demonstrates:
# Generative AI
# Large Language Models (LLMs)
# Tokenization
# Transformer Inference
# Prompt Engineering
# ==========================================

# Install libraries first
# pip install transformers torch

from transformers import pipeline

# Load Generative AI Model

generator = pipeline("text-generation", model="gpt2")

# Define University Knowledge Base

knowledge_base = """
University Helpdesk Information

Hostel Admission Requirements:
- Admission confirmation letter
- Government ID proof
- Two passport size photographs

Course Registration:
- Registration happens online through the university portal
- Students must register before the semester deadline

Fee Payment:
- Fees can be paid through the student dashboard
- Online payment methods include debit card, credit card, and net banking

Examination Schedule:
- Semester exams usually start in December and May

Library Access:
- Students must carry their university ID card
"""

# Chatbot Function

def university_chatbot(question):

    prompt = f"""
You are a university helpdesk assistant.

Answer the student's question clearly using the information below.
Give only relevant information in bullet points.

Knowledge Base:
{knowledge_base}

Student Question: {question}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=40,
        temperature=0.4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )

    return response[0]["generated_text"]

question = input("Ask your question: ")

answer = university_chatbot(question)

print("\nChatbot Response:\n")
print(answer)
