from google import genai
from google.genai import types
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# give title to the page
st.title('AI Chatbot with Google Gemini API')

# initialize session variables at the start once
if 'model' not in st.session_state:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error('GEMINI_API_KEY is missing. Add it to the .env file and restart Streamlit.')
        st.stop()
    st.session_state['model'] = genai.Client(api_key=api_key)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# create sidebar to adjust parameters
st.sidebar.title('Model Parameters')
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider('Max Tokens', min_value=1, max_value=4096, value=256)

# update the interface with the previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# create the chat interface
if prompt := st.chat_input("Enter your query"):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # get response from the model
    with st.chat_message('assistant'):
        client = st.session_state['model']
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in st.session_state['messages']
        ]
        stream = (
            chunk.text
            for chunk in client.models.generate_content_stream(
                model='gemini-3.6-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
        )

        response = st.write_stream(stream)
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # handle message overflow based on the model size