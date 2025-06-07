# Updated app.py
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_huggingface import HuggingFaceEndpoint # type: ignore
import torch
from dotenv import load_dotenv, find_dotenv
from auth_utils import *
import requests
import zipfile
import gdown # type: ignore

# MUST be the first Streamlit command
st.set_page_config(
    page_title="MediBot - Medical Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Professional Medical Color Scheme
st.markdown("""
    <style>
    /* Professional Medical Color Scheme: Blue, Green, White, Soft Gray */

    .auth-container {
        background-color: #f4fafd;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(33, 150, 243, 0.10);
        max-width: 400px;
        margin: 2rem auto;
    }

    .auth-title {
        color: #2196f3;
        text-align: center;
        margin-bottom: 1.8rem;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 0 2px 8px #e3f0ff;
    }

    .auth-input {
        width: 100%;
        padding: 0.95rem;
        border: 1px solid #b3e5fc;
        border-radius: 14px;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
        background: #e3f2fd;
        transition: all 0.3s;
    }

    .auth-input:focus {
        border-color: #2196f3;
        box-shadow: 0 0 0 2px #b2dfdb;
        outline: none;
        background: #e0f7fa;
    }

    .auth-button {
        background: linear-gradient(135deg, #26c6da, #81d4fa);
        color: #0d223a !important;
        border: none;
        border-radius: 14px;
        padding: 1rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px #b2ebf2;
    }

    .auth-button:hover {
        background: linear-gradient(135deg, #29b6f6, #b2ebf2);
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 18px #b2ebf2;
    }

    .auth-switch {
        text-align: center;
        margin-top: 1.5rem;
        color: #2196f3;
        font-weight: 500;
    }

    .auth-switch a {
        color: #2196f3;
        text-decoration: underline;
        font-weight: 600;
        cursor: pointer;
    }

    .error-message {
        color: #e53935;
        background: #fff3f3;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1rem;
        padding: 0.5rem;
        box-shadow: 0 1px 4px #ffd6d6;
    }

    .success-message {
        color: #388e3c;
        background: #e8f5e9;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1rem;
        padding: 0.5rem;
        box-shadow: 0 1px 4px #b9f6ca;
    }
    .source-title {
        font-weight: bold;
        color: #1976d2;
        margin-top: 10px;
    }
    .source-item {
        background-color: #e0f2f1;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 5px 0;
        font-size: 0.85em;
    }
    .header {
        color: #1976d2;
        padding-bottom: 0.5rem;
    }
    .header-title {
    background: linear-gradient(90deg, #00c9a7 0%, #00bcd4 50%, #2196f3 100%);
    color: white; /* Makes the gradient visible without transparency tricks */
    -webkit-background-clip: unset;
    -webkit-text-fill-color: unset;
    background-clip: unset;
    text-fill-color: unset;
    
    padding-bottom: 0.2rem;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 1px;
    text-align: center;
    text-shadow: 0 2px 8px rgba(33, 150, 243, 0.4);
    margin-bottom: 0.5rem;
    line-height: 1.1;
    border-bottom: 3px solid #26c6da;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 2px 10px rgba(0, 188, 212, 0.3);
    background-color: #e0f7fa; /* Light background to help the gradient pop */
}


    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .assistant-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 12px 15px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e0f7fa;
        border-radius: 15px;
        padding: 12px 15px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv(find_dotenv())

DB_PATH = "vectorstore/db_faiss"

# Add this function above get_vectorstore()
def download_vectorstore():
    GDRIVE_FILE_ID = "11tpdXREb7HRKyCeQf4UznqzNKQR3hyH-"
    ZIP_PATH = "vectorstore.zip"

    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        with st.spinner("📥 Downloading medical knowledge base..."):
            # Download zip from Google Drive using gdown
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", ZIP_PATH, quiet=False)
            
            # Extract zip
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(".")

            os.remove(ZIP_PATH)

# Update get_vectorstore()
@st.cache_resource
def get_vectorstore():
    download_vectorstore()
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm(huggingface_repo_id: str, HF_TOKEN: str):
    llm = HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{huggingface_repo_id}",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation"
    )
    return llm

def show_auth_page():
    st.markdown('<h1 class="header-title">🩺 MediBot Authentication</h1>', unsafe_allow_html=True)
    
    # Initialize auth mode
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = "login"
    
    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        if st.session_state.auth_mode == "login":
            st.markdown('<div class="auth-title">Sign in to MediBot</div>', unsafe_allow_html=True)
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username", key="login_username").strip()
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password").strip()
                submit_login = st.form_submit_button("Sign In")
                
                if submit_login:
                    if username and password:
                        success, message = verify_user(username, password)
                        if success:
                            st.session_state.user = username
                            st.session_state.auth_message = "Login successful! Redirecting..."
                            st.session_state.auth_message_type = "success"
                            st.rerun()
                        else:
                            st.session_state.auth_message = message
                            st.session_state.auth_message_type = "error"
                    else:
                        st.session_state.auth_message = "Please fill all fields"
                        st.session_state.auth_message_type = "error"
            
            if 'auth_message' in st.session_state:
                if st.session_state.auth_message_type == "error":
                    st.markdown(f'<div class="error-message">{st.session_state.auth_message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-message">{st.session_state.auth_message}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="auth-switch">Don\'t have an account? </div>', unsafe_allow_html=True)
            
            if st.button("Sign up", key="switch_to_signup"):
                st.session_state.auth_mode = "signup"
                if 'auth_message' in st.session_state:
                    del st.session_state.auth_message
                st.rerun()
        
        elif st.session_state.auth_mode == "signup":
            st.markdown('<div class="auth-title">Create MediBot Account</div>', unsafe_allow_html=True)
            
            with st.form("signup_form"):
                username = st.text_input("Username", placeholder="Choose a username", key="signup_username").strip()
                password = st.text_input("Password", type="password", placeholder="Create a password", key="signup_password").strip()
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm").strip()
                email = st.text_input("Email (optional)", placeholder="Your email address", key="signup_email").strip()
                name = st.text_input("Full Name (optional)", placeholder="Your full name", key="signup_name").strip()
                submit_signup = st.form_submit_button("Create Account")
                
                if submit_signup:
                    if username and password and confirm_password:
                        if password != confirm_password:
                            st.session_state.auth_message = "Passwords do not match"
                            st.session_state.auth_message_type = "error"
                        elif len(password) < 6:
                            st.session_state.auth_message = "Password must be at least 6 characters"
                            st.session_state.auth_message_type = "error"
                        else:
                            success, message = register_user(username, password, email, name)
                            if success:
                                st.session_state.auth_message = message + " Please sign in."
                                st.session_state.auth_message_type = "success"
                            else:
                                st.session_state.auth_message = message
                                st.session_state.auth_message_type = "error"
                    else:
                        st.session_state.auth_message = "Please fill required fields"
                        st.session_state.auth_message_type = "error"
            
            if 'auth_message' in st.session_state:
                if st.session_state.auth_message_type == "error":
                    st.markdown(f'<div class="error-message">{st.session_state.auth_message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-message">{st.session_state.auth_message}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="auth-switch">Already have an account? <a>Sign in</a></div>', unsafe_allow_html=True)
            
            if st.button("Sign in", key="switch_to_login"):
                st.session_state.auth_mode = "login"
                if 'auth_message' in st.session_state:
                    del st.session_state.auth_message
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def medi_bot_app():
    # Sidebar with info
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3305/3305873.png", width=80)
        st.header(f"Welcome, {st.session_state.user}!", divider='blue')
        
        # Add logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            # Clear auth message on logout so it doesn't show up on the auth page
            if 'auth_message' in st.session_state:
                del st.session_state['auth_message']
                if 'auth_message_type' in st.session_state:
                    del st.session_state['auth_message_type']
            st.rerun()
        
        with st.container():
            st.markdown("""
            MediBot is an AI-powered assistant trained on medical literature to help answer your health-related questions.
            """)
            
            st.divider()
            
            st.markdown("""
            <div class="sidebar-section">
                <h4>📌 How it works</h4>
                <ol>
                <li>Ask your medical question</li>
                <li>MediBot searches its knowledge base</li>
                <li>Get evidence-based answers with sources</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sidebar-section">
                <h4>⚠️ Important Disclaimer</h4>
                <p>This is not a substitute for professional medical advice. Always consult with a qualified healthcare provider.</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main header
    st.markdown('<h1 class="header-title">🩺 Ask MediBot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="caption">Your AI-powered medical assistant providing evidence-based answers</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Hello {st.session_state.user}! I'm MediBot, your medical assistant. How can I help you today?",
            "sources": []
        }]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Custom styling for messages
            if message["role"] == "assistant":
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Reference Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)

    # User input
    if prompt := st.chat_input("Type your medical question here..."):
        st.chat_message("user").markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        # Prepare the prompt template
        CUSTOM_PROMPT_TEMPLATE = """
            Use the following pieces of information provided in the context to answer the user's question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Dont provide any additional information that is not in the context.

            Context: {context}
            Question: {question}

            Start your answer directly followed by the answer. No need to repeat the question. No small talk.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            # Show loading spinner
            with st.spinner("🔍 Searching medical knowledge base..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the knowledge base")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]

                # Format sources
                sources = []
                for doc in source_documents:
                    source_name = doc.metadata.get('source', 'Medical Reference')
                    page_content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    sources.append(f"📄 **{source_name}**\n\n{page_content}")

            # Display response
            with st.chat_message("assistant"):
                st.markdown(f'<div class="assistant-message">{result}</div>', unsafe_allow_html=True)
                
                if sources:
                    with st.expander("📚 Reference Sources (Click to expand)", expanded=False):
                        for source in sources:
                            st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)

            # Store in session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result,
                "sources": sources
            })

        except Exception as e:
            st.error(f"⚠️ Error processing your request: {str(e)}")

def main():
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Check authentication status
    if not check_auth_status():
        show_auth_page()
    else:
        medi_bot_app()

if __name__ == "__main__":
    main()