import os
import json
import csv
from PIL import Image
import pandas as pd
import numpy as np
import nltk
import ssl
import random
from datetime import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Open a file to read from it
#this is not possible in my system
#with open('res.json', 'r') as file:
#        file_content = file.read()

intents = [
        {"tag": "greeting",
                "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
                "responses": ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"]
        },
        {"tag": "goodbye",
                "patterns": ["Bye", "See you later", "Goodbye"],
                "responses": ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
        },
        {"tag": "thanks",
                "patterns": ["Thanks", "Thank you", "That's helpful"],
                "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
        {"tag": "BookRoom",
                "patterns": ["I need a room for two nights", "Can I book a room for two nights?", "Can I get a room with a king-sized bed?", "Can I get a room with a balcony?", "Can I get a room with two beds?", "Can I book a room with a garden view?", "Can I get a room with a jacuzzi?"],
                "responses": ["Sure, I can help with that. Which type of room do you prefer?", "Yes, we have rooms with king-sized beds available. Would you like to book one?", "Yes, we have rooms with balconies available. Would you like to book one?", "Yes, we have rooms with two beds available. Would you like to book one?", "Yes, we have rooms with garden views available. Would you like to book one?", "Yes, we have rooms with jacuzzis available. Would you like to book one?"]
        },
        {"tag": "RoomPrice",
                "patterns": ["What is the price of a deluxe room?", "What is the cost of a suite room?"],
                "responses": ["The price of a deluxe room is $200 per night.", "The cost of a suite room is $300 per night."]
        },
        {"tag": "Amenities",
                "patterns": ["Do you have free Wi-Fi?", "Do you have a swimming pool?", "Is there a gym in the hotel?", "Do you have a bar in the hotel?", "Do you have conference facilities?", "Do you have room service?", "Is there a restaurant in the hotel?", "Do you have laundry service?", "Is there a spa in the hotel?", "Do you offer car rental services?", "Do you have a kids' club?", "Do you have a gift shop?", "Do you offer currency exchange services?", "Do you have a concierge service?", "Do you offer babysitting services?"],
                "responses": ["Yes, we offer free Wi-Fi in all rooms and public areas.", "Yes, we have an outdoor swimming pool available for guests.", "Yes, we have a fully equipped gym available for our guests.", "Yes, we have a bar that serves a variety of drinks and snacks.", "Yes, we have conference facilities available for meetings and events.", "Yes, we offer 24-hour room service.", "Yes, we have a restaurant that serves breakfast, lunch, and dinner.", "Yes, we offer laundry and dry cleaning services.", "Yes, we have a spa that offers various treatments and massages.", "Yes, we offer car rental services. Please contact the front desk for more details.", "Yes, we have a kids' club with various activities for children.", "Yes, we have a gift shop that sells souvenirs and other items.", "Yes, we offer currency exchange services at the front desk.", "Yes, we have a concierge service to assist you with your needs.", "Yes, we offer babysitting services. Please contact the front desk for more information."]
        },
        {"tag": "CancelBooking",
                "patterns": ["I want to cancel my reservation"],
                "responses": ["I can help you with that. Can you please provide your booking reference?"]
        },
        {"tag": "MealInclusion",
                "patterns": ["Is breakfast included?", "What time is breakfast served?"],
                "responses": ["Yes, breakfast is included with your stay.", "Breakfast is served from 6:30 AM to 10:00 AM."]
        },
        {"tag": "LateCheckout",
                "patterns": ["Can I check out late?", "Can I get a late check-out?"],
                "responses": ["Late checkout is subject to availability. Please confirm with the front desk.", "Late check-out is subject to availability. Please confirm with the front desk."]
        },
        {"tag": "HotelInfo",
                "patterns": ["What's the address of the hotel?", "What are the check-in and check-out times?", "What is the check-out time?"],
                "responses": ["Our hotel is located at 123 Main Street, City, Country.", "Check-in is from 3 PM and check-out is until 11 AM.", "Check-out time is until 11 AM."]
        },
        {"tag": "Policies",
                "patterns": ["Do you allow pets?", "What is the cancellation policy?"],
                "responses": ["Yes, we are a pet-friendly hotel. Additional charges may apply.", "Our cancellation policy allows free cancellation up to 24 hours before arrival."]
        },
        {"tag": "RoomService",
                "patterns": ["Can I get an extra bed?", "Can I request a wake-up call?", "Do you have room service?"],
                "responses": ["Yes, extra beds are available upon request. Additional charges may apply.", "Yes, we can arrange a wake-up call for you. At what time would you like it?", "Yes, we offer 24-hour room service."]
        },
        {"tag": "Transportation",
                "patterns": ["Do you provide airport shuttle service?", "Do you offer car rental services?"],
                "responses": ["Yes, we offer airport shuttle service. Please provide your flight details to arrange it.", "Yes, we offer car rental services. Please contact the front desk for more details."]
        },
        {"tag": "Dining",
                "patterns": ["Can I make a reservation for dinner?", "What time is breakfast served?"],
                "responses": ["Sure, I can assist with that. For how many people and at what time?", "Breakfast is served from 6:30 AM to 10:00 AM."]
        },
        {"tag": "RoomInfo",
                "patterns": ["What types of rooms do you have?", "Do you have non-smoking rooms?", "Is there a mini-bar in the room?", "Is there a hair dryer in the room?", "Is there a safe in the room?", "Is there a telephone in the room?", "Is there a coffee maker in the room?"],
                "responses": ["We offer standard, deluxe, and suite rooms.", "Yes, we have non-smoking rooms available.", "Yes, all our rooms are equipped with a mini-bar.", "Yes, all our rooms are equipped with a hair dryer.", "Yes, all our rooms are equipped with a safe.", "Yes, all our rooms are equipped with a telephone.", "Yes, all our rooms are equipped with a coffee maker."]
        },
        {"tag": "Payment",
                "patterns": ["Can I pay with a credit card?"],
                "responses": ["Yes, we accept all major credit cards."]
        },
        {"tag": "EarlyCheckin",
                "patterns": ["Can I get an early check-in?"],
                "responses": ["Early check-in is subject to availability. Please confirm with the front desk."]
        }
]
# Prepare data for training
patterns = []
tags = []
for intent in intents:
        for pattern in intent["patterns"]:
                patterns.append(pattern)
                tags.append(intent['tag'])

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X, tags)
image = Image.open('Hotel.png')

# Resize the image
resized_image = image.resize((200, 200))
# Display an image
st.image(resized_image)
st.title("Hotel Chatbot")
st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation")
global response
global user_input
global timestamp
user_input = st.text_input("You: ")
counter = 0
if user_input:
        # Convert input to vector and predict tag
        input_vector = vectorizer.transform([user_input])
        predicted_tag = model.predict(input_vector)[0]
        # Get response from the intents based on the predicted tag
        for intent in intents:
                if intent['tag'] == predicted_tag:
                        response = random.choice(intent['responses'])
        #st.write(f"Bot: {response}")
        st.text_area("Chatbot:",value=response,height=120,max_chars=None,key=f"chatbot_response_{counter}")
timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
#file.close()
counter += 1
def main():
        
        global counter
# Create a sidebar menu with options
        menu= ["Home", "Conversation History", "About","Intents used"]
        choice = st.sidebar.selectbox ("Menu", menu)
# Home Menu
        if choice == "Home":
# Check if the chat_log.csv file exists, and if not, create it with column names
                if not os.path.exists('chat_log.csv'):
                        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
                                counter += 1
                                timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
                                # Save the user input and chatbot response to the chat_log.csv file
                        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow([user_input, response, timestamp])
                        if response.lower() in ['goodbye', 'bye']:
                                st.write("Thank you for chatting with me. Have a great day!")
                                st.stop()
        elif choice == "Conversation History":
                timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
                # Read the conversation history from the CSV file
                st.header("Conversation History")
                with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                        csv_reader = csv.reader (csvfile)
                        print(csv_reader)
#Skip the header row for row in csv_reader:
                        st.text(f"User: {user_input}")
                        if 'response' in globals():
                                st.text(f"Chatbot: {response}")
                        else:
                                st.text("Chatbot: No response available")
                        st.text(f"Timestamp: {timestamp}")
                        st.markdown ("-")
        elif choice == "About":
                st.write("The goal of this project is to create a chatbot that can understand and respond to a conversation")
                st.subheader ("Project Overview:")
                st.write("""
The project is divided into two parts:
1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on la
2. For building the Chatbot interface, Streamlit web framework is used to build a web-
""")

                st.subheader("Dataset:")
                st.write("""
The dataset used in this project is a collection of labelled intents and entities. The - Intents: The intent of the user input (e.g. "greeting", "booking slots", "about")
- Entities: The entities extracted from user input (e.g. "Hi", "is it really safe to eat here?","Could you tell your COVID safety protocols?"- Text: The user input text.
""")
                st.subheader("Streamlit Chatbot Interface:")
                st.write("The chatbot interface is built using Streamlit.")
                st.subheader("Conclusion: ")
                st.write("In this project, a chatbot is built that can understand and respond to users")
        
        elif choice=="Intents used":
                st.subheader("Intents are:")
                df = pd.DataFrame(intents)
                st.write(df[['tag', 'patterns']])

if __name__=='__main__':
        main()
