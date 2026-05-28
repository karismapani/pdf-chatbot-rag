import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set your API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# Load and split PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"✅ PDF loaded! {len(chunks)} chunks created.")
    return chunks

# Create vector store
def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created!")
    return vectorstore

# Ask question
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below.
    Context: {context}
    Question: {question}
    """)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer

# Main
if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ")
    chunks = load_pdf(pdf_path)
    vectorstore = create_vectorstore(chunks)
    
    print("\n🤖 PDF ChatBot Ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = ask_question(vectorstore, question)
        print(f"Bot: {answer}\n")
