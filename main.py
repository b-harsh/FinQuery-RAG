import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def run_groq_engine():
    if not os.path.exists("report.pdf"):
        print(" Error: report.pdf not found in root folder!")
        return
        
    print(" Loading PDF...")
    loader = PyPDFLoader("report.pdf")
    docs = loader.load_and_split(CharacterTextSplitter(chunk_size=1000, chunk_overlap=100))

    print(" Building Local Memory (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-l6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    print(" Connecting to Groq (Llama-3)...")
    model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    template = """Answer based ONLY on the following context:
    {context}
    
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print("\n Groq Financial Engine Active!")
    while True:
        query = input("\nYour Question: ")
        if query.lower() in ['exit', 'quit']: break
        
        print(" Searching...")
        try:
            response = chain.invoke(query)
            print(f"\nAI: {response}")
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    run_groq_engine()