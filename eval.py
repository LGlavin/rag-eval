from dotenv import load_dotenv

load_dotenv()

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from openai import OpenAI
from ragas.embeddings import OpenAIEmbeddings

from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

questions = [
    "What is DiffMAS?",
    "What benchmarks did DiffMAS improve on?",
    "How does latent communication work in DiffMAS?",
]

samples = []
for q in questions:
    response = query_engine.query(q)
    samples.append(
        {
            "user_input": q,
            "response": str(response),
            "retrieved_contexts": [str(n.node.text) for n in response.source_nodes],
        }
    )

dataset = EvaluationDataset.from_list(samples)

faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy(embeddings=OpenAIEmbeddings(client=OpenAI()))
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(results)
