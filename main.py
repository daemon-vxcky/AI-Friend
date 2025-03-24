import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import sqlite3

# Load Pretrained Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)  # Positive, Neutral, Negative

def analyze_sentiment(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    scores = F.softmax(output.logits, dim=1).numpy()[0]
    sentiment = ["Negative", "Neutral", "Positive"][scores.argmax()]
    return sentiment, scores

def save_chat(user_input, bot_response, sentiment):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS chats (user TEXT, bot TEXT, sentiment TEXT)")
    c.execute("INSERT INTO chats (user, bot, sentiment) VALUES (?, ?, ?)", (user_input, bot_response, sentiment))
    conn.commit()
    conn.close()

def main():
    st.title("My AI Friend - Emotional Support Chatbot")
    user_input = st.text_input("Type your message:")
    
    if st.button("Send") and user_input:
        sentiment, scores = analyze_sentiment(user_input)
        bot_response = "I'm here for you!" if sentiment == "Negative" else "That sounds great!"
        save_chat(user_input, bot_response, sentiment)
        
        st.write(f"**AI Friend:** {bot_response}")
        st.write(f"**Sentiment:** {sentiment}")

    if st.button("View Chat History"):
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("SELECT * FROM chats")
        rows = c.fetchall()
        conn.close()
        for row in rows:
            st.write(f"**User:** {row[0]}")
            st.write(f"**AI:** {row[1]}")
            st.write(f"**Sentiment:** {row[2]}")
            st.write("---")

if __name__ == "__main__":
    main()