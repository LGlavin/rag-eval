from dotenv import load_dotenv
load_dotenv()

import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate

qa_prompt = PromptTemplate(
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}\n"
    "Only answer using the context above. "
    "if the answer is not in the context, say 'I don't have that information'"
    "Query: {query_str}\n"
    "Answer: "
)

if os.path.exists("./storage"):
    print("Loading from disk...")
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
else:
    print("Building index...")
    reader = SimpleDirectoryReader(input_files=["2604.21794v1.pdf"])
    documents = reader.load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50 )
    nodes = splitter.get_nodes_from_documents(documents) 
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir="./storage")


query_engine = index.as_query_engine()
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
print("RAG ready. Type 'quit' to exit. \n")

while True:
    question = input("Ask a question: ")
    if question.lower() == 'quit':
        break
    response = query_engine.query(question)
    print(f"\n{response}\n")


