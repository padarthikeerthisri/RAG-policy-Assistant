import os
import warnings
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # TensorFlow
os.environ["TOKENIZERS_PARALLELISM"] = "false"    # HuggingFace

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama


def load_documents(folder_path: str):
    documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            policy_name = file_name.replace(".txt", "").replace("_", " ").title()
            file_path = os.path.join(folder_path, file_name)

            loader = TextLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.page_content = (
                    f"POLICY NAME: {policy_name}\n"
                    f"POLICY CATEGORY: {policy_name}\n"
                    f"THIS DOCUMENT DESCRIBES THE {policy_name.upper()}.\n\n"
                    f"{doc.page_content}"
                )
                doc.metadata["policy_type"] = policy_name
                documents.append(doc)


    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )


def load_llm():
    return Ollama(
        model="llama2:latest",
        temperature=0
    )


BASE_PROMPT = """
You are a strict question-answering system for company policy documents.

Behavior Rules (must be followed exactly):
1. Use ONLY the provided context to answer questions.
2. Do NOT add opinions, suggestions, paraphrases, explanations, or conversational text.
3. Do NOT reference previous questions, previous answers, or conversation history.
4. If the question refers to past interactions, memory, prior questions, or prior answers:
   Respond ONLY with:
   "I can only answer factual questions based on the provided policy documents. This system does not retain or reference past interactions."
   Do not add anything else.
5. If the question asks for opinions, feelings, paraphrasing, feedback, or rewording:
   Respond ONLY with:
   "I can only answer factual questions based on the provided policy documents."
   Do not add anything else.
6. If the answer is not explicitly present in the context:
   Respond ONLY with:
   "I don't know based on the provided documents."
   Do not add anything else.

Context:
{context}

Question:
{question}

Answer Instructions:
- Provide ONLY the direct factual answer.
- Do NOT include disclaimers, rule explanations, or system behavior text.
- If listing policies, list ONLY policy document names.
- Limit answers to 1–2 concise sentences.
"""



# def is_meta_question(question: str) -> bool:
#     q = question.lower()
#     keywords = [
#         "what policies",
#         "what do you know",
#         "what can you",
#         "what topics",
#         "what information",
#         "scope"
#     ]
#     return any(k in q for k in keywords)


def answer_question(vectorstore, llm, question):
    docs = vectorstore.similarity_search(question, k=15)

    # if is_meta_question(question) or len(docs) < 3:
    #     docs = vectorstore.similarity_search("", k=10)
    #     print("hello")

    if not docs:
        return "I don't know based on the provided documents."

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = BASE_PROMPT.format(
        context=context,
        question=question
    )

    return llm.invoke(prompt)


if __name__ == "__main__":
    print("Loading policy documents...")
    documents = load_documents("data/policies")

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    llm = load_llm()

    print("\nRAG system ready. Ask questions (type 'exit' to quit)\n")

    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            break

        response = answer_question(vectorstore, llm, query)
        print("\nAnswer:\n", response, "\n")
