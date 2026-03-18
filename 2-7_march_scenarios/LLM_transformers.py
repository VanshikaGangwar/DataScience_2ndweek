
# In[1]:
#  Scenario: AI‑Powered Email Assistant
# You’re building an email drafting assistant for a corporate team.
# - When a user starts typing:
# “Artificial Intelligence will…”
# - The assistant uses GPT‑2 to auto‑complete the sentence with a coherent continuation.
# - Example output:
# “Artificial Intelligence will transform the way businesses interact with customers,
# enabling faster decisions and personalized experiences.”

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

print(generator("Artificial Intelligence will", max_length=30))

# In[2]:
#  Scenario: Game Storyline Generator
# You’re designing a fantasy role‑playing game and need quick ideas for quests and dialogue.
# - The developer types a prompt:
# “The hero enters the ancient forest and discovers…”
# - The GPT‑2 pipeline continues the story:
# “The hero enters the ancient forest and discovers a hidden village where the people whisper of
#  a cursed artifact buried beneath the old ruins.”

from transformers import pipeline

# Load GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")

prompt = "The hero enters the ancient forest and discovers"

result = generator(
    prompt,
    max_length=80,      # maximum length of generated text
    num_return_sequences=1,  # number of generated outputs
    temperature=0.8     # randomness (0.7–1.0 is good for creativity)
)

# Print generated story
print(result[0]["generated_text"])


# In[3]:
# Scenario: Research Assistant for Academic Papers
# Imagine you’re building a tool for university researchers who analyze large collections of academic papers.
# - A researcher uploads a sentence:
# “Deep learning models are powerful”
# - Your system uses the BERT tokenizer to break the sentence into smaller units (tokens).

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Deep learning models are powerful"

tokens = tokenizer.tokenize(text)

print(tokens)

# In[4]:
# 🏥 Scenario: Healthcare Chatbot for Patient Queries
# A hospital is developing a chatbot to help patients ask questions about their health.
# - A patient types:
# “Deep learning models are powerful” (or in practice, something like “I have chest pain and shortness of breath”).
# - The chatbot uses the BERT tokenizer to break the sentence into tokens:

from transformers import BertTokenizer

# Load pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Patient query
text = "I have chest pain and shortness of breath"

# Tokenize the sentence
tokens = tokenizer.tokenize(text)

print("Tokens:", tokens)



# In[5]:
# Scenario: Exploring Creativity in AI Writing
# A university professor is running a workshop on creative writing with AI.
# - The professor gives students the prompt:
# “The future of Artificial Intelligence”
# - The system uses GPT‑2 to generate continuations of the sentence.
# To demonstrate how temperature affects creativity:
# - With temperature = 0.2 (low randomness), the output is more predictable and conservative, e.g.:
# “The future of Artificial Intelligence will bring advancements in healthcare, education, and business.”

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of Artificial Intelligence"

print(generator(prompt, max_length=40, temperature=0.2))
print(generator(prompt, max_length=40, temperature=0.8))

# In[6]:
#  Scenario: Scriptwriting Assistant for Film Production
# A film studio is experimenting with AI to help writers brainstorm dialogue and plot ideas.
# - The writer provides the prompt:
# “The future of Artificial Intelligence”
# - The system generates two versions of the continuation:
# - Temperature = 0.2 (low randomness)
# Output is more predictable and formal, e.g.:
# “The future of Artificial Intelligence will improve industries, enhance productivity, and reshape education.”
# → This is useful for serious, factual narration in a documentary script.
# - Temperature = 0.8 (higher randomness)
# Output is more imaginative and varied, e.g.:
# “The future of Artificial Intelligence dances with uncertainty, weaving stories of machines dreaming and societies reborn.”
# → This is perfect for creative dialogue or speculative sci‑fi storytelling

from transformers import pipeline

# Load GPT-2 text generation model
generator = pipeline("text-generation", model="gpt2")

# Writer's prompt
prompt = "The future of Artificial Intelligence"

# Low temperature (more predictable)
low_temp_output = generator(
    prompt,
    max_length=50,
    temperature=0.2,
    num_return_sequences=1
)

# High temperature (more creative)
high_temp_output = generator(
    prompt,
    max_length=50,
    temperature=0.8,
    num_return_sequences=1
)

print("Temperature = 0.2 (Formal Output)")
print(low_temp_output[0]["generated_text"])

print("\nTemperature = 0.8 (Creative Output)")
print(high_temp_output[0]["generated_text"])


# In[7]:
# Scenario: AI‑Powered News Headline Generator
# A digital media company wants to help journalists brainstorm headlines and opening lines for articles.
# - The system asks the journalist to enter a prompt (e.g., “Enter a prompt: The future of Artificial Intelligence”).
# - The GPT‑2 model then generates a continuation of the text, up to 50 tokens.
# - Example output:
# “The future of Artificial Intelligence will reshape industries, redefine creativity, and challenge
#  our understanding of human potential.”

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = input("Enter a prompt: ")

output = generator(prompt, max_length=50)

print(output[0]["generated_text"])

# In[8]:
# 🎓 Scenario: Student Brainstorming Tool for Essay Writing
# A college student is struggling to start their essay.
# - The system prompts them:
# “Enter a prompt:”
# - The student types:
# “The impact of technology on education”
# - The GPT‑2 model generates a continuation up to 50 tokens, for example:
# “The impact of technology on education has transformed classrooms, enabling online learning,
# personalized study plans, and global collaboration among students.”

from transformers import pipeline

# Load GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Ask student for prompt
prompt = input("Enter a prompt: ")

# Generate essay continuation
result = generator(
    prompt,
    max_length=50,          # generate up to 50 tokens
    num_return_sequences=1, # one output
    temperature=0.7         # balanced creativity
)

# Display generated text
print("\nGenerated Essay Idea:\n")
print(result[0]['generated_text'])


# In[9]:
# Scenario: AI Model Evaluation in a Startup
# A startup is building an AI‑powered content assistant to help marketers draft product descriptions.
# - The team wants to compare two models: GPT‑2 (larger, more expressive) and DistilGPT‑2 (smaller, faster, distilled version).
# - They use the same prompt:
# “Artificial Intelligence will transform”
# - The system generates two outputs:
# - GPT‑2 Output
# Longer, more detailed continuation, e.g.:
# “Artificial Intelligence will transform industries by automating processes, enhancing creativity, and
# reshaping how humans interact with technology.”
# → Richer but slower to run.
# - DistilGPT‑2 Output


from transformers import pipeline

gpt2 = pipeline("text-generation", model="gpt2")
distilgpt2 = pipeline("text-generation", model="distilgpt2")

prompt = "Artificial Intelligence will transform"

print("GPT2:", gpt2(prompt, max_length=40))
print("DistilGPT2:", distilgpt2(prompt, max_length=40))

# In[10]:
# Scenario: Educational Tool for Classroom Learning
# A high school teacher is introducing students to AI‑powered writing assistants.
# - The teacher wants to show how different models produce different styles of text.
# - They use the same prompt:
# “Artificial Intelligence will transform”
# - The system generates two outputs:
# - GPT‑2 Output
# More elaborate continuation, e.g.:
# “Artificial Intelligence will transform the way societies function, influencing education, healthcare, and the future of human creativity.”
# → Great for showing students how AI can generate rich, detailed ideas.
# - DistilGPT‑2 Output
# Shorter, faster continuation, e.g.:
# “Artificial Intelligence will transform our daily lives.”
# → Useful for demonstrating concise summaries.

from transformers import pipeline

# Prompt given by teacher
prompt = "Artificial Intelligence will transform"

# GPT-2 model
gpt2_generator = pipeline("text-generation", model="gpt2")

gpt2_output = gpt2_generator(
    prompt,
    max_length=40,
    num_return_sequences=1
)

# DistilGPT-2 model
distilgpt2_generator = pipeline("text-generation", model="distilgpt2")

distil_output = distilgpt2_generator(
    prompt,
    max_length=40,
    num_return_sequences=1
)

# Print results
print("GPT-2 Output:\n")
print(gpt2_output[0]["generated_text"])

print("\nDistilGPT-2 Output:\n")
print(distil_output[0]["generated_text"])
