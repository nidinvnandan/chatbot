import streamlit as st
from chat import response,nlp,model,data,intents,words,classes

# ERROR_THRESHOLD
ERROR_THRESHOLD = 0.25

# Streamlit App
st.title("Chatbot")

# Counter for generating unique keys

user_input = st.text_input('Enter the Question')
if st.button("Submit"):
        st.text("Chatbot:")
        response_text = response(user_input)
        st.text(response_text)
        
    
