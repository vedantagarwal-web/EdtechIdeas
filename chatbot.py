import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai_api_key

# Load and preprocess the college essay dataset
essay_files = ['essay1.txt', 'essay2.txt', ...]  # List of essay file paths
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = []
for essay_file in essay_files:
    loader = TextLoader(essay_file)
    documents = loader.load()
    docs.extend(documents)
texts = text_splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Create retriever and RAG chain
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4')
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

# Streamlit app
st.title("College Essay Chatbot")

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Enter your essay prompt:", key='user_input')

if user_input:
    # Generate essay using RAG chain
    essay = rag_chain.run(user_input)

    # Add user prompt and generated essay to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": essay})

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'**You:** {message["content"]}')
    else:
        st.markdown(f'**Assistant:** {message["content"]}')