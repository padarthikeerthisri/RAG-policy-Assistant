from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from main import (
    load_documents,
    chunk_documents,
    create_vectorstore,
    load_llm,
    answer_question
)

app = FastAPI(title="RAG Policy Assistant")
templates = Jinja2Templates(directory="templates")


class QueryRequest(BaseModel):
    question: str


documents = load_documents("data/policies")
chunks = chunk_documents(documents)
vectorstore = create_vectorstore(chunks)
llm = load_llm()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
def ask(request: QueryRequest):
    answer = answer_question(vectorstore, llm, request.question)
    return {"answer": answer}
