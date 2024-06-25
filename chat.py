# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import streamlit as st
# # from langchain.llms import Ollama
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_groq import ChatGroq
# from langchain.docstore.document import Document

# st.set_page_config(
#     page_title="ChatBot",
#     page_icon=":robot_face:",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
# st.header("Chat Bot")

# from PyPDF2 import PdfReader

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# pdf_text = ""
# if uploaded_file is not None:
#     pdf_reader = PdfReader(uploaded_file)
    
#     for page in pdf_reader.pages:
#         pdf_text += page.extract_text()


# # reader = PyPDFLoader(uploaded_file)
# # doc = reader.load()

#     documents = [Document(page_content=pdf_text)]

#     input_text = st.text_input("Enter Question")

#     submit = st.button("Generate")


#     text_splitter = RecursiveCharacterTextSplitter(chunk_size =400,chunk_overlap=100)
#     text = text_splitter.split_documents(documents)

#     db = FAISS.from_documents(text,OllamaEmbeddings())

#     # db.similarity_search("Explain YOLO")

#     # llm = Ollama(model = "llama2", temperature = 0.5)
#     llm = ChatGroq(groq_api_key="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",model = "mixtral-8x7b-32768", temperature = 0.5)

#     prompt = ChatPromptTemplate.from_template(
#             """
#     Answer the following question based only on the provided context. 
#     Think step by step before providing a detailed answer.  
#     <context>
#     {context}
#     </context>
#     Question: {input}"""
#     )

#     document_chain = create_stuff_documents_chain(llm,prompt)
#     retriever = db.as_retriever()


#     retrieval_chain = create_retrieval_chain(retriever,document_chain)

#     # response = retrieval_chain.invoke({"input":"What are different types of detection algorithms"})

#     # response["answer"]

#     if submit:
#         response = retrieval_chain.invoke({"input":input_text})
#         st.write(response["answer"])

# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_groq import ChatGroq
# from streamlit.runtime.caching import cache_resource
# from langchain.agents.agent_toolkits import create_python_agent
# from langchain.agents import load_tools, initialize_agent
# from langchain.agents import AgentType
# from langchain.tools.python.tool import PythonREPLTool
# from langchain.python import PythonREPL
# from langchain.chat_models import ChatOpenAI

# st.set_page_config(
#     page_title="ChatBot",
#     page_icon=":robot_face:",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
# st.header("Chat Bot")

# @st.cache_data
# def load_pdf(file):
#     pdf_reader = PdfReader(file)
#     pdf_text = ""
#     for page in pdf_reader.pages:
#         pdf_text += page.extract_text()
#     return pdf_text

# @st.cache_data
# def create_faiss_database(_documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
#     text = text_splitter.split_documents(_documents)
#     db = FAISS.from_documents(text, OllamaEmbeddings())
#     return db

# @cache_resource
# def load_llm():
#     llm = ChatGroq(groq_api_key="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", model="mixtral-8x7b-32768", temperature=0.5)
#     return llm

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# if uploaded_file is not None:
#     pdf_text = load_pdf(uploaded_file)
#     documents = [Document(page_content=pdf_text)]

#     input_text = st.text_input("Enter Question")
#     submit = st.button("Generate")

#     if submit and input_text:
#         db = create_faiss_database(documents)
#         llm = load_llm()
#         prompt = ChatPromptTemplate.from_template(
#             """
#             Answer the following question based only on the provided context. 
#             Think step by step before providing a detailed answer.  
#             <context>
#             {context}
#             </context>
#             Question: {input}"""
#         )
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = db.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         response = retrieval_chain.invoke({"input":input_text})
#         st.write(response["answer"])


# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_groq import ChatGroq
# from streamlit.runtime.caching import cache_resource
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.agent_toolkits.load_tools import load_tools

# st.set_page_config(
#     page_title="ChatBot",
#     page_icon=":robot_face:",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
# st.header("Chat Bot")

# @st.cache_data
# def load_pdf(file):
#     pdf_reader = PdfReader(file)
#     pdf_text = ""
#     for page in pdf_reader.pages:
#         pdf_text += page.extract_text()
#     return pdf_text

# @st.cache_data
# def create_faiss_database(_documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
#     text = text_splitter.split_documents(_documents)
#     db = FAISS.from_documents(text, OllamaEmbeddings())
#     return db

# @cache_resource
# def load_llm():
#     llm = ChatGroq(groq_api_key="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", model="mixtral-8x7b-32768", temperature=0.5)
#     return llm

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# if uploaded_file is not None:
#     pdf_text = load_pdf(uploaded_file)
#     documents = [Document(page_content=pdf_text)]

#     input_text = st.text_input("Enter Question")
#     submit = st.button("Generate")

#     if submit and input_text:
#         db = create_faiss_database(documents)
#         llm = load_llm()
#         retriever = db.as_retriever()
#         retriever_tool = create_retriever_tool(retriever, "Uploaded Documents",
#                       "Search for information about 2D Object Detection. For any questions about the content in the uploaded document, you must use this tool!")
#         tools = load_tools(["wikipedia"], llm=llm) + [retriever_tool]
#         agent = initialize_agent(tools, llm, 
#             agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#             handle_parsing_errors=True,
#             verbose=True)
        
#         result = agent.run(input_text)
#         st.write(result)


# import os
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_groq import ChatGroq
# from streamlit.runtime.caching import cache_resource
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain_community.llms import HuggingFaceHub

# st.set_page_config(
#     page_title="ChatBot",
#     page_icon=":robot_face:",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
# st.header("Chat Bot")

# @st.cache_data
# def load_pdf(file):
#     pdf_reader = PdfReader(file)
#     pdf_text = ""
#     for page in pdf_reader.pages:
#         pdf_text += page.extract_text()
#     return pdf_text

# @st.cache_data
# def create_faiss_database(_documents, index_path):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
#     text = text_splitter.split_documents(_documents)
#     db = FAISS.from_documents(text, OllamaEmbeddings())
#     db.save_local(index_path)
#     return db

# @st.cache_data
# def load_faiss_database(index_path):
#     if os.path.exists(index_path):
#         db = FAISS.load_local(index_path, OllamaEmbeddings(), allow_dangerous_deserialization=True)
#     else:
#         db = None
#     return db

# @cache_resource
# def load_llm():
# #     hf=HuggingFaceHub(
# #     repo_id="mistralai/Mistral-7B-v0.1",
# #     model_kwargs={"temperature":0.1,"max_length":500},
# #     huggingfacehub_api_token = "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"

# # )
#     llm = ChatGroq(groq_api_key="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", model="mixtral-8x7b-32768", temperature=0.5)
#     return llm

# uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
# if uploaded_files:
#     documents = []
#     index_paths = []
    
#     for uploaded_file in uploaded_files:
#         pdf_text = load_pdf(uploaded_file)
#         documents.append(Document(page_content=pdf_text))
#         index_path = os.path.join("faiss_indexes", f"faiss_index_{uploaded_file.name}")
#         index_paths.append(index_path)
    
#     context = st.text_input("Enter the context in which to use the database tool")
#     input_text = st.text_input("Enter Question")
#     submit = st.button("Generate")

#     if submit and input_text and context:
#         llm = load_llm()
#         all_retrievers = []

#         for doc, index_path in zip(documents, index_paths):
#             db = load_faiss_database(index_path)
#             if db is None:
#                 db = create_faiss_database([doc], index_path)
#             retriever = db.as_retriever()
#             all_retrievers.append(create_retriever_tool(retriever, f"Document: {index_path}",
#                       f"Search for information about the content in the uploaded document: {index_path}. For any questions about the content in the uploaded document, you must use this tool when the context is: {context}"))

#         tools = load_tools(["wikipedia"], llm=llm) + all_retrievers
#         agent = initialize_agent(tools, llm, 
#             agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#             handle_parsing_errors=True,
#             verbose=True)
        
#         result = agent.run(input_text)
#         st.write(result)

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from streamlit.runtime.caching import cache_resource
from langchain.agents import initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.llms import HuggingFaceHub

st.set_page_config(
    page_title="ChatBot",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.header("Chat Bot")

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

        tools = load_tools(["wikipedia"], llm=llm) + all_retrievers
        agent = initialize_agent(tools, llm, 
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True)
        
        result = agent.run(input_text)
        st.write(result)


