# BookSearchArchive
My code to create a PostgreSQL / pgvector based archive of a collection of books I want to search (lexically and semantically). 
Also includes an interface with Large Language Models (LLMs) for answering questions about the books via
natural language queries. These natural language queries are used in a Retrieval Augmented Generation (RAG)
pipeline to generate answers.

The current stack is all open-source and includes:
- PostgreSQL
- pgvector
- Hugging Face Transformers
- Haystack


