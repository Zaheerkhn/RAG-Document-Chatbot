import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Initialize Model & Embeddings
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
embeddings = NVIDIAEmbeddings()

# Streamlit UI
st.title("Langchain : RAG Application")
st.subheader("Upload your Document and ask anything about it.")

# Sidebar: Upload PDF
st.sidebar.subheader("📂 Upload PDF & Generate Embeddings")
session_id = st.sidebar.text_input("Session ID", value="Default_session")
pdf_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Process PDFs and Generate Embeddings
if st.sidebar.button("Generate Embeddings"):
    try:
        if pdf_files:
            documents = []
            temp_pdf_paths = []
            
            with st.spinner("📖 Processing PDFs..."):
                for pdf_file in pdf_files:
                    temp_path = f'./temp_{pdf_file.name}'
                    temp_pdf_paths.append(temp_path)
                    with open(temp_path, "wb") as file:
                        file.write(pdf_file.read())

                for path in temp_pdf_paths:
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    documents.extend(docs)
                    os.remove(path)  # Cleanup temp files

            # Text Splitting and Vector Store Creation
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)

            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            # Contextualize Question
            contextualize_q_system_prompt = """Given a chat history and the latest user question, 
            reformulate it into a standalone question without referencing past chat history."""
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Answer Generation
            qa_system_prompt = """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Keep the answer short, concise and simple to understand.
            
            Context: {context}
            """ 
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Attach Message History
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            st.session_state.rag_chain = conversational_rag_chain
            st.success("✅ Embeddings Generated Successfully! Ask your questions now.")

        else:
            st.warning("⚠️ Please upload at least one PDF file.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Chat Interface
if "rag_chain" in st.session_state:
    user_input = st.chat_input("Ask anything about your docs")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

    for msg in st.session_state.messages[-10:]:  # Keep last 10 messages
        with st.chat_message(msg["role"].lower()):
            st.write(msg["content"])

    if user_input:
        user_input = user_input[:300]  # Truncate user input to 300 characters
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        session_history = get_session_history(session_id)

        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

        with st.chat_message("assistant"):
            st.write(response['answer'])

        with st.expander("📜 Chat History"):
            for msg in session_history.messages[-10:]:  # Show last 10 messages
                if msg.type == "human":
                    st.markdown(f"**You:** {msg.content}")
                else:
                    st.markdown(f"**Assistant:** {msg.content}")
