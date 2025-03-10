# LangChain: RAG Application

## 📌 Overview

LangChain: RAG Application is a **Retrieval-Augmented Generation (RAG)**-based chatbot that allows users to upload PDFs, generate embeddings, and ask questions related to the documents. It leverages **NVIDIA AI endpoints**, **FAISS for vector storage**, and **LangChain** for retrieval and response generation.

## 🚀 Features

- **PDF Upload**: Users can upload multiple PDF documents.
- **Embedding Generation**: Converts document text into vector embeddings using NVIDIAEmbeddings.
- **Contextual Question Understanding**: Reformulates user queries based on chat history.
- **Intelligent Document Retrieval**: Uses FAISS for retrieving relevant document chunks.
- **Conversational Memory**: Maintains session-based chat history.
- **Real-time Q&A**: Provides accurate responses using an LLM (Llama 3.3 70B-Instruct).

## 🏗️ Tech Stack

- **Python**
- **Streamlit** (UI framework)
- **LangChain** (Text processing & retrieval)
- **FAISS** (Vector store)
- **NVIDIA AI Endpoints** (LLM & Embeddings)

## 📥 Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Zaheerkhn/RAG-Document-Chatbot
   cd rag-app
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file and add:
     ```env
     NVIDIA_API_KEY=your_nvidia_api_key
     ```

## 🎯 Usage

1. Run the application:
   ```sh
   streamlit run app.py
   ```
2. **Upload PDFs** in the sidebar.
3. Click **Generate Embeddings** to process documents.
4. **Ask questions** about the uploaded documents in the chat.

## 📌 Workflow

1. **PDF Loading**: Documents are loaded using `PyPDFLoader`.
2. **Text Splitting**: Content is split into smaller chunks for efficient retrieval.
3. **Vectorization**: FAISS stores embeddings for fast similarity searches.
4. **Question Reformulation**: Converts user queries into a standalone format.
5. **Retrieval & Answer Generation**: The system fetches relevant document sections and generates responses.
6. **Session Management**: Chat history is stored per session for contextual responses.

## 🤖 Model Used

- **Meta Llama 3.3 70B-Instruct** (via NVIDIA AI Endpoints) for generating responses.
- **FAISS** for document retrieval.
- **NVIDIA NIM** for Inferencing.

## 🛠️ Future Enhancements

- **Multimodal Support** (Processing images & tables from PDFs)
- **Fine-tuned models for better accuracy**
- **UI/UX improvements**

## 📜 License

This project is licensed under the **Apache License 2.0**.

