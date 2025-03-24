import sqlite3
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BlenderbotTokenizer, BlenderbotForConditionalGeneration
import datetime
import random

# Load emotion classifier
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Load improved conversational model (Blenderbot instead of DialoGPT)
# Blenderbot tends to be more engaging and personable than DialoGPT
model_name = "facebook/blenderbot-400M-distill"
chatbot_tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
chatbot_model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Database setup
def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            message TEXT,
            response TEXT,
            emotion TEXT,
            activity TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Function to store chat history
def store_chat(user, message, response, emotion, activity):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chats (user, message, response, emotion, activity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
              (user, message, response, emotion, activity, str(datetime.datetime.now())))
    conn.commit()
    conn.close()

# Function to classify emotion
def classify_emotion(user_input):
    emotion_result = classifier(user_input)[0]
    return emotion_result["label"]

# Personality templates to make responses more lively based on emotion
personality_templates = {
    "joy": [
        "That's fantastic! {}",
        "I'm so happy to hear that! {}",
        "Wonderful! {} Let's keep that positive energy going!",
        "That's awesome! {} What else has been making you happy?"
    ],
    "sadness": [
        "I understand how you feel. {}",
        "It's okay to feel down sometimes. {} Things will get better.",
        "I'm here for you. {}",
        "That sounds tough. {} Remember to be kind to yourself during difficult times."
    ],
    "anger": [
        "I can see why that would be frustrating. {}",
        "It's understandable to feel that way. {} Let's try to work through this.",
        "That would upset me too. {}",
        "I hear your frustration. {} Would it help to talk more about what happened?"
    ],
    "fear": [
        "It's okay to feel anxious about that. {}",
        "Many people would feel the same way. {}",
        "That sounds concerning. {} Let's think about this together.",
        "I understand your worry. {} What would help you feel more secure?"
    ],
    "surprise": [
        "Wow! {} That's unexpected!",
        "I didn't see that coming either! {}",
        "That's quite a surprise! {}",
        "Really? {} Tell me more about how that happened!"
    ],
    "love": [
        "That's so heartwarming! {}",
        "I'm touched hearing that. {}",
        "What a beautiful sentiment! {}",
        "Love is such a special feeling. {}"
    ]
}

# Enhanced response generation using Blenderbot with emotion-based personalization
def generate_response(user_input, emotion, activity, _):
    # We don't use chat_history_ids with Blenderbot the same way as DialoGPT
    # Instead we'll focus on making each response more contextually relevant
    
    # Create emotionally appropriate prompt
    inputs = chatbot_tokenizer(user_input, return_tensors="pt")
    
    # Generate base response
    reply_ids = chatbot_model.generate(**inputs, max_length=100)
    base_response = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    # Personalize the response based on detected emotion
    templates = personality_templates.get(emotion, ["{}"])
    template = random.choice(templates)
    personalized_response = template.format(base_response)
    
    # Sometimes add the activity suggestion into the response
    if random.random() < 0.3:  # 30% chance to include activity
        activity_phrases = [
            f"By the way, have you considered this? {activity}",
            f"You might enjoy: {activity}",
            f"Something that might help: {activity}",
            f"A suggestion for you: {activity}"
        ]
        personalized_response += " " + random.choice(activity_phrases)
    
    # Return personalized response (and None for compatibility with your existing code)
    return personalized_response, None

# Define enhanced activity suggestions based on emotions
activity_suggestions = {
    "joy": [
        "Share your happiness with a friend or family member!", 
        "Engage in a creative hobby like painting or playing an instrument.",
        "Go outside for a nature walk and enjoy the moment.",
        "Listen to an upbeat playlist and dance along!"
    ],
    "sadness": [
        "Listen to soothing music like 'Weightless' by Marconi Union.",
        "Try journaling your thoughts and feelings to process them.",
        "Watch a feel-good movie or read an uplifting book like 'The Alchemist'.",
        "Practice mindfulness meditation using an app like Headspace."
    ],
    "anger": [
        "Try deep-breathing exercises—inhale for 4 seconds, hold for 7, exhale for 8.",
        "Engage in a physical activity like jogging or yoga to release tension.",
        "Write down your thoughts and then rip the paper to symbolically let go.",
        "Listen to calming nature sounds or white noise to relax."
    ],
    "fear": [
        "Talk to someone you trust about your feelings and get reassurance.",
        "Write down what's worrying you and challenge negative thoughts.",
        "Try a short guided meditation for anxiety relief.",
        "Do a grounding exercise: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste."
    ],
    "surprise": [
        "Reflect on why this surprise happened and embrace the excitement!",
        "Share the news with friends or family and celebrate.",
        "Write about your experience in a journal to remember it.",
        "Explore something new related to the surprise, like learning a related skill."
    ],
    "love": [
        "Express your feelings through a heartfelt message or letter.",
        "Plan a meaningful activity with someone you care about.",
        "Read a romantic novel or watch a feel-good movie about love.",
        "Practice self-love: Treat yourself to something that makes you happy."
    ]
}

def get_activity_suggestions(emotion):
    return activity_suggestions.get(emotion, ["I'm here to chat with you!"])

# Function to fetch chat history
def fetch_chat_history(user):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("SELECT timestamp, user, message, response, emotion, activity FROM chats WHERE user = ? ORDER BY timestamp DESC", (user,))
    chats = c.fetchall()
    conn.close()
    return chats