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


This code is related to my Artificial Intelligence blog posts. 
* [The Book Search Archive Intro Post] (https://www.mindfiretechnology.com/blog/archive/our-open-source-ai-stack-the-book-search-archive/)
* [How to Setup your environment] (https://www.mindfiretechnology.com/blog/archive/environment-setup-for-rag-using-python-haystack-postgresql-pgvector-and-hugging-face/)
* [List of all Artificial Intelligence blog posts] (https://www.mindfiretechnology.com/blog/categories/Artificial%20Intelligence?p=2)
