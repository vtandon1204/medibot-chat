# 1. Setup LLM (Mistral with HuggingFace)

import os
from langchain_huggingface import HuggingFaceEndpoint # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_community.llms import HuggingFaceHub # type: ignore

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3" 

def load_llm(huggingface_repo_id: str):
    llm =  HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{huggingface_repo_id}",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation"  # Explicitly specify task
    )   
    return llm

# 2. Connect LLM with the vector database (Faiss)

CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of information provided in the context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Dont provide any additional information that is not in the context.

Context: {context}
Question: {question}

Start your answer with "Answer: " directly followed by the answer. No other text before "Answer: ". No need to repeat the question. No small talk.

"""


def set_custom_prompt(custom_prompt_template):
    """
    Set the custom prompt template.
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

from langchain_community.vectorstores import FAISS  # type: ignore
import torch

DB_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)


# 3. Create a chain to answer questions

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 4. Test the chain with a sample question
import traceback

user_query = input("Enter your question: ")

try:
    retrieved_docs = db.similarity_search(user_query, k=3)
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i + 1}:\n{doc.page_content}\n")

    response = qa_chain.invoke({"query": user_query})
    print(f"{response['result']}")
    for i, doc in enumerate(response['source_documents']):
        print(f"\nSource #{i + 1}:\n{doc.page_content}")
except Exception as e:
    print("An error occurred while generating the answer:", str(e))
    traceback.print_exc()
