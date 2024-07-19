# text-rag
Retrieval Augmented Generation from a specified path containing pdf and txt files.

Compatible only with HuggingFace local models.

To get started
### Step 1: Create an instance of the `RAGPipeline` class
```python
import rag_pipeline

rag = rag_pipeline.RAGPipeline(model_name="microsoft/Phi-3-mini-4k-instruct", directory='files', max_new_tokens=1000)
```
In this example, we create an instance of the `RAGPipeline` class, specifying the `directory` parameter as `'middlebury'`, the model as phi-3 and the `max_new_tokens` parameter as `1000`.

### Step 2: Define the query
```python
query = "What is the penalty function used in the evaluation?"
```

### Step 3: Run the RAG pipeline
```python
rag.rag_pipeline(query)
```
Finally, we call the `rag_pipeline` method on the `rag` object, passing the query as an argument. This will generate a RAG pipeline for the given query.