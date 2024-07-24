# Text-RAG
Retrieval Augmented Generation from a specified path containing pdf and txt files.

Compatible only with HuggingFace local models working on Nvidia gpu's.


**Install Packages**

To install the required packages, run the following command in your terminal:
```bash
pip install numpy==1.24.1 langchain-community==0.2.7 langchain==0.2.9 transformers==4.42.4  bitsandbytes==0.43.2 flash-attn==2.6.2 sentence-transformers==3.0.1 faiss-cpu flash_attention pypdf
```
You also need to install cuda compatible pytorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To get started
### Step 1: Create an instance of the `RAGPipeline` class
```python
from text_rag import RAGPipeline

rag = RAGPipeline(model_name="microsoft/Phi-3-mini-4k-instruct", directory='your_directory', max_new_tokens=1000)
```
In this example, we create an instance of the `RAGPipeline` class, specifying the `directory` parameter as `'your_directory'`, the model as phi-3 and the `max_new_tokens` parameter as `1000`.

Make sure that your directory contains at least one pdf or txt file. If not either add one or modify the code to read whatever your file type is in prepare documents part.

### Step 2: Create an index
```python
rag.create_index()
```

### Step 3: Define the query
```python
query = "What is the penalty function used in the evaluation?"
```

### Step 4: Run the RAG pipeline
```python
rag(query)
```
Finally, we call the `rag_pipeline` method on the `rag` object, passing the query as an argument. This will generate a RAG pipeline for the given query.

## You can also load an index!

```python
# Initialize RAGPipeline
rag = RAGPipeline(directory='your_directory', max_new_tokens=500)

# Load the saved index
rag.load_index('path_to_saved_index')

# Run
rag(query)
```