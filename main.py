import streamlit as st
#importing the funtions and training from chat.py file
from chat import response,nlp,model,data,intents,words,classes

# ERROR_THRESHOLD
ERROR_THRESHOLD = 0.25

# Streamlit App
st.title("Chatbot")



user_input = st.text_input('Enter the Question(example, machine learning,nlp,deep learning,tell me a joke etc)')
if st.button("Submit"):
        st.text("Chatbot:")
        #here the response function will be imported from the chat.py file which contaims the training part
        response_text = response(user_input)
        st.write(response_text)
        
    
