from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings, )
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
import requests
from bs4 import BeautifulSoup

chat_history = []
text = ''


# Get contents in webpage from url with library
def updateText():
    # Get text data from url
    url = "https://textdoc.co/fCAmzT1RyWtlN9qj"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(strip=True)  # Get all text and strip whitespace
    text = text.replace(
        "Online Text Editor - Create, Edit, Share and Save Text FilesTextdocZipdocWriteurlTxtshareOnline CalcLoading…Open FileSave to Drive",
        "")
    text = text.replace("/ Drive Add-on", "")
    return text

# Invoke QA chain upon user submission from client side
def on_submit(query):

    global chat_history, text

    # Refresh text
    text = ''
    text = updateText()
    print("updated text ", text[0:100])

    # Create new tokenizer and text splitter
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=24,
        length_function=lambda x: len(tokenizer.encode(x)),
    )

    # Create new chunks and embeddings
    chunks = text_splitter.create_documents([text])
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    # Create new Chroma DB and QA chain
    db = Chroma.from_documents(chunks, embedding_function)
    qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1),
                                                     db.as_retriever())

    # Process the query with fresh QA chain
    result = qa_chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result["answer"]
