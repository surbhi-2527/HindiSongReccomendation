# ===============================
# IMPORTS
# ===============================
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# ===============================
# LOAD SONG DATA
# ===============================
df = pd.read_csv("songs.csv")
df["combined"] = df["title"] + " " + df["artist"] + " " + df["mood"]

print("Loading embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Encoding songs...")
song_embeddings = embed_model.encode(df["combined"].tolist())


# ===============================
# LOAD LLM (Phi-2)
# ===============================
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=["songs", "user_input"],
    template="""
User mood or input: {user_input}

Recommended songs:
{songs}

Explain why these songs match the user's mood in simple and emotional language.
"""
)

explanation_chain = LLMChain(llm=llm, prompt=prompt)


# ===============================
# SONG RECOMMENDATION FUNCTION
# ===============================
def recommend_songs(input_text, top_n=4):
    query_embedding = embed_model.encode([input_text])
    similarities = cosine_similarity(query_embedding, song_embeddings)[0]
    top_idx = similarities.argsort()[::-1][:top_n]

    recommendations = []
    for i in top_idx:
        recommendations.append(
            f"- {df.loc[i, 'title']} : {df.loc[i, 'mood']}"
        )

    return recommendations


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("\n=== Hindi Song Recommendation System ===\n")
    user_input = input("Enter a song name / mood / lyric: ")

    # Get recommendations
    songs = recommend_songs(user_input)

    # Print like screenshot
    print("\nRecommended Hindi Songs:")
    for song in songs:
        print(song)

    # LLM explanation
    explanation = explanation_chain.run(
        songs="\n".join(songs),
        user_input=user_input
    )

    print("\nExplain why these songs match the user's mood:\n")
    print(explanation.strip())

    # Static demo section (for project / viva)
    print("\nSongs that play at the same time:\n")
    print("- I love the beautiful woman")
    print("- I'm so happy for you")
    print("- I wanted to look at you in the mirror")
