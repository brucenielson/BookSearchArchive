from __future__ import annotations
import wikipedia
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.generativeai.types import Tool, FunctionDeclaration
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerateContentResponse
from doc_retrieval_pipeline import DocRetrievalPipeline
# noinspection PyPackageRequirements
from haystack import Document
from generator_model import get_secret
from llm_message_utils import send_message
from typing import Dict, Any

# Load secrets & configure Gemini
gemini_key = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')
genai.configure(api_key=gemini_key)

# Initialize your document retriever
doc_retriever = DocRetrievalPipeline(
    table_name="popper_archive",
    db_user_name="postgres",
    db_password=get_secret(r'D:\Documents\Secrets\postgres_password.txt'),
    postgres_host="localhost",
    postgres_port=5432,
    db_name="postgres",
    verbose=False,
    llm_top_k=5,
    retriever_top_k_docs=100,
    use_reranker=True,
    embedder_model_name="BAAI/llm-embedder",
)

# Create the chat session
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat(history=[])


def format_doc(doc: Document) -> str:
    """Format a Haystack Document into a simple string."""
    lines = [doc.content]
    for k, v in doc.meta.items():
        if not k.startswith("_"):
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def search_datastore(datastore_query: str) -> Dict[str, str]:
    """Tool: search your local document store."""
    docs, _ = doc_retriever.generate_response(datastore_query, min_score=0.6)
    if not docs:
        return {"result": "No matching documents found."}
    snippets = [format_doc(d) for d in docs[:3]]
    return {"result": "\n\n".join(snippets)}


def search_wikipedia(wiki_page_search: str, question: str) -> Dict[str, str]:
    """Tool: fallback to Wikipedia if needed."""
    try:
        page = wikipedia.page(wiki_page_search, auto_suggest=False)
        return {"result": page.summary}
    except Exception as e:
        return {"result": f"Error fetching Wikipedia: {e}"}


def answer(final_answer: str) -> Dict[str, str]:
    """Tool: provide the final answer and terminate."""
    return {"result": f"FINAL ANSWER: {final_answer}"}


tools = [
    Tool(function_declarations=[
        FunctionDeclaration(
            name="search_datastore",
            description="Query the local document store.",
            parameters={
                "type": "object",
                "properties": {"datastore_query": {"type": "string"}},
                "required": ["datastore_query"],
            },
        ),
        FunctionDeclaration(
            name="search_wikipedia",
            description="Search Wikipedia and return its summary.",
            parameters={
                "type": "object",
                "properties": {
                    "wiki_page_search": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["wiki_page_search", "question"],
            },
        ),
        FunctionDeclaration(
            name="answer",
            description="Return the final answer.",
            parameters={
                "type": "object",
                "properties": {"final_answer": {"type": "string"}},
                "required": ["final_answer"],
            },
        ),
    ])
]


def run(question: str):
    # Kick things off
    prompt = (
        "You can call search_datastore or search_wikipedia to gather information. "
        "When you know the answer, call answer(final_answer).\n\n"
        f"Question: {question}"
    )
    response: GenerateContentResponse = send_message(chat, prompt, tools=tools)

    # Single-pass: handle one function call or plain response
    candidate = response.candidates[0]
    for part in candidate.content.parts:
        # If it's plain text, print it
        if getattr(part, "text", None):
            print(part.text.strip())

        # If it's a function call, dispatch it immediately
        if getattr(part, "function_call", None):
            func = part.function_call
            name = func.name
            # Unpack the proto args into a plain dict
            raw_args = func.args or {}
            pretty_args = {}
            for k, v in raw_args.items():
                if hasattr(v, "WhichOneof"):
                    kind = v.WhichOneof("kind")
                    pretty_args[k] = getattr(v, kind)
                else:
                    # Already a basic Python type (str, int, etc)
                    pretty_args[k] = v

            print(f"\n[Calling {name} with args {pretty_args}]")

            # Execute the Python implementation
            # Dispatcher mapping function names to actual functions
            dispatch = {
                "search_wikipedia": search_wikipedia,
                "search_datastore": search_datastore,
                "answer": answer,
            }

            # Validate and call
            if name not in dispatch:
                raise ValueError(f"Unknown function: {name}")

            func = dispatch[name]
            response_data = func(**pretty_args)
            result = response_data["result"]
            print(f"[Result]: {result}\n")

            # If this was the final answer, done
            if name == "answer":
                return

            # Otherwise send the result back to Gemini and print its reply
            followup = send_message(chat, result, tools=tools)
            print(followup.candidates[0].content.parts[0].text.strip())
            return


if __name__ == "__main__":
    run("Is induction valid in some cases, particularly when doing statistics?")
