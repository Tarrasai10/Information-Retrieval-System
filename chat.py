import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from streamlit.runtime.caching import cache_resource
from langchain.agents import initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.load_tools import load_tools
import faiss
import wikipediaapi

st.set_page_config(
    page_title="Q-A System",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.header("Information Retrieval System")

@st.cache_data
def load_pdf(file):
    pdf_reader = PdfReader(file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

@st.cache_data
def create_faiss_database(_documents, index_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    text = text_splitter.split_documents(_documents)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(text, embeddings)
    db.save_local(index_path)
    return db

@st.cache_data
def load_faiss_database(index_path):
    if os.path.exists(index_path):
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        db = None
    return db

@cache_resource
def load_llm():
    llm = ChatGroq(groq_api_key="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", model="mixtral-8x7b-32768", temperature=0.5)
    return llm

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    index_paths = []
    
    for uploaded_file in uploaded_files:
        pdf_text = load_pdf(uploaded_file)
        documents.append(Document(page_content=pdf_text))
        index_path = os.path.join("faiss_indexes", f"faiss_index_{uploaded_file.name}")
        index_paths.append(index_path)
    
    context = st.text_input("Enter the context in which to use the database tool")
    input_text = st.text_input("Enter Question")
    submit = st.button("Generate")

    if submit and input_text and context:
        llm = load_llm()
        all_retrievers = []

        for doc, index_path in zip(documents, index_paths):
            db = load_faiss_database(index_path)
            if db is None:
                db = create_faiss_database([doc], index_path)
            retriever = db.as_retriever()
            all_retrievers.append(create_retriever_tool(retriever, f"Document: {index_path}",
                      f"Search for information about the content in the uploaded document: {index_path}. For any questions about the content in the uploaded document, you must use this tool when the context is: {context}"))

        tools = all_retrievers

        try:
            wikipedia_tool = load_tools(["wikipedia"], llm=llm)
            tools += wikipedia_tool
        except ImportError as e:
            st.error(f"Error loading Wikipedia tool: {e}")

        agent = initialize_agent(tools, llm, 
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True)
        
        result = agent.run(input_text)
        st.write(result)

