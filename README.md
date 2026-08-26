# Corrective RAG: Six-Phase Implementation

This project implements a Corrective Retrieval-Augmented Generation (CRAG) pipeline using LangGraph, Mistral AI, FAISS, and Tavily Search.

## Workflow

```text
Question
   ↓
1. Document Retrieval
   ↓
2. Document Evaluation
   ↓
3. Routing Decision
   ├── CORRECT → Context Refinement
   └── INCORRECT / AMBIGUOUS
          ↓
      4. Query Rewriting
          ↓
      5. Web Search
          ↓
      Context Refinement
          ↓
6. Answer Generation
```

## Technologies

- Python
- LangGraph
- LangChain
- Mistral AI
- FAISS
- Tavily Search
- PyPDF
- Pydantic
- dotenv

## Installation

```bash
pip install langchain langgraph langchain-community langchain-mistralai langchain-text-splitters faiss-cpu pypdf tavily-python python-dotenv
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Place the source PDF in the project directory:

```text
Corrective RAG/
├── 6_ambiguous.ipynb
├── book1.pdf
├── README.md
└── .env
```

## Six Phases

### Phase 1: Document Retrieval

The PDF is loaded, split into overlapping chunks, embedded using Mistral embeddings, and stored in a FAISS vector database.

```python
docs = PyPDFLoader("./book1.pdf").load()
chunks = splitter.split_documents(docs)

embedding = MistralAIEmbeddings(model="mistral-embed")
vector_store = FAISS.from_documents(chunks, embedding)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

The retriever returns the four most relevant document chunks for the question.

### Phase 2: Document Evaluation

Each retrieved chunk is evaluated by the language model.

The evaluator returns:

- A relevance score between `0.0` and `1.0`
- A short explanation

The thresholds are:

```python
UPPER_TH = 0.7
LOWER_TH = 0.3
```

Classification rules:

| Condition | Verdict |
|---|---|
| Any score greater than `0.7` | `CORRECT` |
| All scores less than `0.3` | `INCORRECT` |
| Otherwise | `AMBIGUOUS` |

Relevant documents with scores above `0.3` are stored in `good_docs`.

### Phase 3: Routing

The graph chooses the next step based on the evaluation verdict.

- `CORRECT`: use the retrieved documents
- `INCORRECT`: search the web
- `AMBIGUOUS`: search the web because the retrieved context may be incomplete

```python
def route_after_eval(state):
    if state["verdict"] == "CORRECT":
        return "refine"
    return "rewrite_query"
```

### Phase 4: Query Rewriting

For `INCORRECT` and `AMBIGUOUS` results, the original question is rewritten into a concise web-search query.

Example:

```text
Batch normalization vs layer normalization
```

Possible rewritten query:

```text
batch normalization versus layer normalization differences
```

The rewritten query is stored in `web_query`.

### Phase 5: Web Search and Context Refinement

Tavily searches for external information using the rewritten query.

The web results are converted into LangChain `Document` objects.

Next, the context is:

1. Decomposed into sentences
2. Evaluated sentence by sentence
3. Filtered to keep only relevant sentences
4. Recombined into `refined_context`

This prevents unrelated information from being passed to the answer generator.

### Phase 6: Answer Generation

The final answer is generated using only the refined context.

The model is instructed to respond with:

```text
I don't know.
```

when the context is empty or insufficient.

This reduces unsupported or hallucinated answers.

## State Structure

The LangGraph state contains:

```python
class State(TypedDict):
    question: str
    docs: List[Document]

    good_docs: List[Document]
    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_docs: List[Document]
    web_query: str

    answer: str
```

## Graph Structure

The LangGraph pipeline is:

```text
START
  ↓
retrieve
  ↓
eval_each_doc
  ├── CORRECT → refine
  └── INCORRECT / AMBIGUOUS
          ↓
      rewrite_query
          ↓
      web_search
          ↓
        refine
          ↓
       generate
          ↓
        END
```

## Running the Notebook

1. Open `6_ambiguous.ipynb` in Visual Studio Code.
2. Select the Python kernel.
3. Add your API keys to `.env`.
4. Ensure `book1.pdf` is available.
5. Run the cells from top to bottom.
6. Execute the final cell.

Example question:

```python
"Batch normalization vs layer normalization"
```

The final output displays:

- Retrieval verdict
- Evaluation reason
- Rewritten web query
- Generated answer

## Purpose of CRAG

Traditional RAG always trusts retrieved documents. CRAG improves reliability by evaluating the retrieved context first.

If the context is:

- Reliable, it is refined and used directly.
- Irrelevant, external web search is performed.
- Partially relevant, web search supplements the missing information.

This creates a more robust retrieval and generation pipeline.
