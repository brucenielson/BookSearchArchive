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
    lines = []
    for k, v in doc.meta.items():
        if not k.startswith("_"):
            lines.append(f"{k}: {v}")
    lines.extend([doc.content])
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
    ])
]


def run(question: str):
    prompt = (
        "You may call search_datastore and then search_wikipedia to gather information. "
        "Preferably call both. Then answer the question.\n\n"
        f"Question: {question}"
    )
    print(f"Prompt: {prompt}\n")
    response: GenerateContentResponse = send_message(chat, prompt, tools=tools)

    while True:
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            # If it's plain text, print it
            if getattr(part, "text", None):
                print("Response:")
                print(part.text.strip())

            # If it's a function call, dispatch it
            if getattr(part, "function_call", None):
                func = part.function_call
                name = func.name
                raw_args = func.args or {}
                pretty_args = {}

                for k, v in raw_args.items():
                    if hasattr(v, "WhichOneof"):
                        kind = v.WhichOneof("kind")
                        pretty_args[k] = getattr(v, kind)
                    else:
                        pretty_args[k] = v

                print(f"\n[Calling {name} with args {pretty_args}]")

                dispatch = {
                    "search_wikipedia": search_wikipedia,
                    "search_datastore": search_datastore,
                }

                if name not in dispatch:
                    raise ValueError(f"Unknown function: {name}")

                result = dispatch[name](**pretty_args)["result"]
                result = result.strip() + ("\n Have you called both functions yet? "
                                           "If not, please do so before answering.")
                print(f"[Result]: {result}\n")

                response = send_message(chat, result, tools=tools)
                break  # Break for-loop to handle next full response
        else:
            # No function calls left; exit the outer loop
            break


if __name__ == "__main__":
    run("Is induction valid in some cases, particularly when doing statistics?")
