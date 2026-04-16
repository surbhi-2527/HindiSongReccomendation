🎵 AI-Based Hindi Song Recommendation System

An intelligent Hindi Song Recommendation System that suggests songs based on a user’s mood, song name, or lyrics using semantic similarity and explains the recommendations using a generative AI model.
 
This project demonstrates the integration of Information Retrieval and Generative AI concepts in a practical application.

📌 Features

🔍 Mood / text-based Hindi song recommendation

🧠 Semantic understanding using sentence embeddings

📊 Cosine similarity for ranking relevant songs

🤖 AI-generated explanation for why songs match the user’s mood

🖥️ Simple terminal-based interface (easy to extend to web)

🛠️ Technologies Used

Python

Pandas – data handling

Sentence Transformers – semantic embeddings

Cosine Similarity (scikit-learn) – ranking

Hugging Face Transformers – generative model

LangChain – LLM integration

🧩 Models Used
Purpose	Model
Text Embeddings	paraphrase-multilingual-MiniLM-L12-v2
Text Generation	microsoft/phi-2
🧠 How It Works

Song data (title, artist, mood) is combined into a single text field

Semantic embeddings are generated for all songs

User input is converted into an embedding

Cosine similarity finds the most relevant songs

An LLM explains why the recommendations fit the user’s mood
