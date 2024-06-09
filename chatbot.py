import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import os
import numpy as np

# Load OpenAI API key from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set the OpenAI API key in the .env file or environment variable 'OPENAI_API_KEY'.")

# Initialize OpenAI LLM
llm = ChatOpenAI(temperature=0.7, model_name='gpt-4')

# Load and preprocess the college essay dataset
essay_files = ['essay1.txt', 'essay2.txt', 'essay3.txt', 'essay4.txt','essay5.txt', 'essay6.txt', 'essay7.txt', 'essay8.txt','essay9.txt']  # List of essay file paths
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

# Create retriever
retriever = vectorstore.as_retriever()

# Debate configurations
max_rounds = 5
num_debaters = 3

# Topic generation prompt
topic_prompt = PromptTemplate(
    input_variables=["user_background"],
    template="""Generate a unique and thought-provoking essay topic based on the user's background and experiences:
    {user_background}
    
    The essay topic should be creative, intellectual, and encourage divergent thinking. It should inspire an essay that is either humorous, emotional, or demonstrates exceptional creativity and intellectual vitality.
    """
)

# Debater prompts
debater_prompt = PromptTemplate(
    input_variables=["topic", "retrieved_documents", "user_background"],
    template="""You are a debater in a multi-agent debate. The debate topic is: {topic}
    
    Use the following retrieved documents as inspiration and supporting evidence for your arguments:
    {retrieved_documents}
    
    Also, consider the user's background and experiences:
    {user_background}
    
    Present your arguments and engage in the debate. Remember to:
    - Take a clear stance on the topic
    - Provide well-reasoned arguments supported by evidence
    - Incorporate the user's background and experiences when relevant
    - Use humor, emotion, or demonstrate exceptional creativity and intellectual vitality
    - Anticipate and counter opposing arguments
    - Maintain a respectful and constructive tone
    """
)

# Judge prompts
judge_prompt_discriminative = PromptTemplate(
    input_variables=["debate_history"],
    template="""You are the judge in a multi-agent debate. Based on the debate history:
    {debate_history}
    
    Determine if a satisfactory resolution has been reached. Respond with "Yes" if a resolution has been reached, or "No" if the debate should continue.
    """
)

judge_prompt_extractive = PromptTemplate(
    input_variables=["debate_history", "user_background"],
    template="""You are the judge in a multi-agent debate. Based on the debate history:
    {debate_history}
    
    Extract the most compelling arguments from the debate to form a cohesive and unforgettable 600-word essay. The essay should incorporate insights from the retrieved documents, the debaters' arguments, and the user's background:
    {user_background}
    
    The essay should be either humorous, emotional, or demonstrate exceptional creativity and intellectual vitality. It should leave a lasting impression on the reader.
    
    Ensure the essay has a clear introduction, body paragraphs addressing key points, and a conclusion.
    """
)

# Streamlit app
st.title("Unique Essay Generator")

# User background questions
st.subheader("Tell us about your background")
user_background = []
user_background.append(st.text_input("Describe your academic achievements and challenges:", key='academic_background'))
user_background.append(st.text_input("Share a significant personal experience that has shaped your perspective:", key='personal_experience'))
user_background.append(st.text_input("What are your career aspirations and why?", key='career_aspirations'))
user_background_text = "\n".join(user_background)

if st.button("Generate Essay") and all(user_background):
    try:
        # Generate essay topic
        essay_topic = llm.predict(topic_prompt.format(user_background=user_background_text))

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(essay_topic)
        retrieved_docs_text = "\n".join([doc.page_content for doc in retrieved_docs])

        # Initialize debaters and judge
        debaters = [ChatOpenAI(temperature=0.7, model_name='gpt-4') for _ in range(num_debaters)]
        judge = ChatOpenAI(temperature=0.7, model_name='gpt-4')

        # Debate loop
        debate_history = []
        for round in range(max_rounds):
            for i, debater in enumerate(debaters):
                # Generate argument as embeddings
                argument_embeddings = debater.predict(debater_prompt.format(topic=essay_topic, retrieved_documents=retrieved_docs_text, user_background=user_background_text))
                debate_history.append(f"Debater {i+1}: {argument_embeddings}")

            # Judge's discriminative evaluation
            debate_history_text = "\n".join(debate_history)
            resolution_reached = judge.predict(judge_prompt_discriminative.format(debate_history=debate_history_text))
            if resolution_reached.lower() == "yes":
                break

        # Judge's extractive summarization
        debate_history_text = "\n".join(debate_history)
        essay = judge.predict(judge_prompt_extractive.format(debate_history=debate_history_text, user_background=user_background_text))

        # Display the generated essay
        st.subheader("Generated Essay")
        st.write(essay)
    except Exception as e:
        st.error(f"An error occurred: {e}")