import os
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgbpython 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import sklearn.metrics.pairwise as pw

@st.cache_data
def load_data_and_model():
    try:
        df = pd.read_csv("cricketers.csv")
    except FileNotFoundError:
        df = pd.DataFrame({
            "Name": ["Virat Kohli", "Steve Smith", "Babar Azam", "Joe Root"],
            "Country": ["India", "Australia", "Pakistan", "England"],
            "Date_Of_Birth": ["1988-11-05", "1989-06-02", "1994-10-15", "1990-12-30"],
            "Test": [113, 107, 52, 140],
            "ODI": [295, 155, 117, 146],
            "T20": [117, 67, 104, 32]
        })

    df['Test'] = df['Test'].fillna(0)
    df['ODI'] = df['ODI'].fillna(0)
    df['T20'] = df['T20'].fillna(0)
    df['Total_Matches'] = df['Test'] + df['ODI'] + df['T20']

    team_experience = df.groupby("Country")["Total_Matches"].sum().reset_index()
    team_experience.rename(columns={"Total_Matches": "Squad_Experience"}, inplace=True)
    
    if team_experience.empty:
        team_experience = pd.DataFrame({"Country": ["India", "Australia", "England", "Pakistan"], "Squad_Experience": [5000, 4800, 4600, 4500]})

    exp_map = team_experience.set_index("Country")["Squad_Experience"].to_dict()

    model_filename = "cricket_win_model_with_csv.pkl"
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        np.random.seed(42)
        n_samples = 2500
        countries = team_experience["Country"].tolist()
        
        team1_list = np.random.choice(countries, n_samples)
        team2_list = np.random.choice(countries, n_samples)

        df_matches = pd.DataFrame({
            "team1": team1_list,
            "team2": team2_list,
            "toss_winner": np.random.randint(0, 2, n_samples),
            "batting_first": np.random.randint(0, 2, n_samples),
            "venue_avg_score": np.random.normal(172, 12, n_samples),
            "current_rrr": np.random.uniform(6.0, 14.0, n_samples),
        })

        df_matches["team1_exp"] = df_matches["team1"].map(exp_map).fillna(5000)
        df_matches["team2_exp"] = df_matches["team2"].map(exp_map).fillna(5000)

        prob_win = 1 / (1 + np.exp(-((df_matches["team1_exp"] - df_matches["team2_exp"]) / 5000 + (df_matches["toss_winner"] * 0.1))))
        df_matches["team1_won"] = (np.random.rand(n_samples) < prob_win).astype(int)
        
        X = df_matches[["toss_winner", "batting_first", "venue_avg_score", "current_rrr", "team1_exp", "team2_exp"]]
        y = df_matches["team1_won"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_filename)

    return df, model, exp_map

df, model, exp_map = load_data_and_model()


# Rag

@st.cache_resource
def init_knowledge_base():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [
        "The Duckworth-Lewis-Stern (DLS) method is a mathematical formulation designed to calculate target scores in rain-interrupted matches.",
        "India won their historic first Cricket World Cup in 1983 under the captaincy of Kapil Dev, defeating the West Indies at Lord's.",
        "Sachin Tendulkar holds the record for the most international runs scored across all formats with over 34,000 runs.",
        "T20 International matches restrict each team to a single innings of a maximum of 20 overs, lasting roughly 3 hours.",
        "Test cricket is played over five days between major international full member nations recognized by the ICC."
    ]
    embeddings = embed_model.encode(texts)
    return embed_model, texts, embeddings

embed_model, knowledge_texts, text_embeddings = init_knowledge_base()

def semantic_search(query, k=2):
    query_emb = embed_model.encode([query])
    scores = pw.cosine_similarity(query_emb, text_embeddings)[0]
    top_indices = scores.argsort()[::-1][:k]
    return [knowledge_texts[i] for i in top_indices]


# 

def get_player_stats(player_name):
    result = df[df['Name'].str.lower() == player_name.lower()]
    if result.empty:
        return None
    row = result.iloc[0]
    return (
        f"**Profile for {row['Name']}:**\n"
        f"- **Country:** {row['Country']}\n"
        f"- **Date of Birth:** {row['Date_Of_Birth'] if pd.notna(row['Date_Of_Birth']) else 'Unknown'}\n"
        f"• Test Matches: {int(row['Test'])}\n"
        f"• ODI Matches: {int(row['ODI'])}\n"
        f"• T20 Matches: {int(row['T20'])}"
    )

def get_match_prediction(team1: str, team2: str):
    t1_exp = exp_map.get(team1, 5000)
    t2_exp = exp_map.get(team2, 5000)

    features = np.array([[1, 1, 175.0, 8.5, t1_exp, t2_exp]])
    probabilities = model.predict_proba(features)[0]
    team1_prob = round(probabilities[1] * 100, 2)
    team2_prob = round(probabilities[0] * 100, 2)
    predicted_winner = team1 if team1_prob > team2_prob else team2

    return (
        f" **Match Prediction: {team1} vs {team2}**\n"
        f"- **Predicted Winner:** {predicted_winner}\n"
        f"- **{team1} Win Probability:** {team1_prob}%\n"
        f"- **{team2} Win Probability:** {team2_prob}%"
    )

def handle_query(user_input):
    text = user_input.lower()
    
    # 1. Check Player Stats
    for player in df['Name']:
        if player.lower() in text:
            stats = get_player_stats(player)
            if stats: return stats

    # 2. Check Match Prediction
    if "predict" in text or "vs" in text or "win" in text:
        countries = df['Country'].dropna().unique()
        found_countries = [c for c in countries if c.lower() in text]
        if len(found_countries) >= 2:
            return get_match_prediction(found_countries[0], found_countries[1])

    # 3. Fallback to Semantic Knowledge Search
    docs = semantic_search(user_input)
    if docs:
        context = "\n".join([f"- {doc}" for doc in docs])
        return f" **RAG Knowledge Assistant:**\nBased on your query, here is what I found:\n{context}"

    return "I couldn't quite understand that. Try asking for a player's stats, predicting a match between two countries, or asking a cricket rule!"


# --- 4. STREAMLIT WEB INTERFACE ---

st.set_page_config(page_title="Cricket RAG & ML Chatbot", page_icon="🏏", layout="centered")

st.title("🏏 Cricket Assistant & RAG Chatbot")
st.markdown("Your AI assistant powered by **Machine Learning (Predictions)**, **Structured Data (Stats)**, and **Semantic Search (RAG)**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about a player, predict a match, or ask a rule..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    response = handle_query(user_query)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)