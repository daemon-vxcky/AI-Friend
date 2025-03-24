import streamlit as st
from functions import classify_emotion, generate_response, get_activity_suggestions, store_chat, fetch_chat_history, init_db

# Initialize database
init_db()

st.title("AI Friend - Emotion-Based Chatbot")
user = st.text_input("Enter your username:", "Guest")
user_input = st.text_area("You:")

# Initialize chat history if not set
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = None

if st.button("Send") and user_input:
    # Emotion detection
    emotion = classify_emotion(user_input)

    # Retrieve activity suggestions
    activities = get_activity_suggestions(emotion)
    activity = activities[0] if activities else "No suggestion available."

    # Generate chatbot response with enhanced input
    response, st.session_state['chat_history'] = generate_response(user_input, emotion, activity, st.session_state['chat_history'])

    # Store chat in database
    store_chat(user, user_input, response, emotion, activity)
    
    # Display chatbot response and activity suggestions
    st.write(f"**Bot ({emotion} detected):** {response}")
    st.write(f"**Suggested Activity:** {activity}")

# Display chat history
if st.button("Show Chat History"):
    chats = fetch_chat_history(user)
    for chat in chats:
        st.write(f"[{chat[0]}] **{chat[1]}:** {chat[2]}")
        st.write(f"➡ **Bot ({chat[4]} detected):** {chat[3]}")
        st.write(f"📝 **Suggested Activity:** {chat[5]}")
        st.write("---")
