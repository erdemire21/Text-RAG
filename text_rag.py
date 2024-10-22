import os
from urllib.request import urlretrieve
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import bitsandbytes as bnb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", directory='files', max_new_tokens=500, chunk_size = 256, chunk_overlap = 50, k_chunks = 5):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.retrievalQA = None
        self.directory = directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_chunks = k_chunks
        self.load_model()
        
    def prepare_documents(self, directory=None):
        if directory is None:
            directory = self.directory
            
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap

        pdf_docs = []
        txt_docs = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    pdf_docs.extend(loader.load())
                elif filename.lower().endswith('.txt'):
                    loader = TextLoader(file_path)
                    txt_docs.extend(loader.load())
                else:
                    warnings.warn(f"Unsupported file type: {filename}. Skipping.")
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        all_docs = pdf_docs + txt_docs

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs_after_split = text_splitter.split_documents(all_docs)
        return docs_after_split
    
    # can also use dunzhang/stella_en_400M_v5
    def create_embeddings(self, model_name="sentence-transformers/all-MiniLM-l6-v2"):
        if self.embeddings is None:
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={'device':'cuda', 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self.embeddings

    def create_vectorstore(self, docs):
        if self.vectorstore is None:
            embeddings = self.create_embeddings()
            self.vectorstore = FAISS.from_documents(docs, embeddings)
        return self.vectorstore

    def save_index(self, path):
        if self.vectorstore is not None:
            self.vectorstore.save_local(path)
            print(f"Index saved at {path}")
        else:
            print("No vectorstore found. Create the vectorstore first using `create_vectorstore`.")

    def load_index(self, path):
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(path, self.create_embeddings(), allow_dangerous_deserialization=True)
            print(f"Index loaded from {path}")
        else:
            print("Vectorstore already exists. Please ensure you are loading the correct index.")



    def is_ampere_gpu():
        """Check if the system has an NVIDIA Ampere or later GPU."""
        if not torch.cuda.is_available():
            logging.info("CUDA is not available.")
            return False
        
        cmd = "nvidia-smi --query-gpu=name --format=csv,noheader"
        try:
            output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            gpu_names = output.strip().split("\n")
            
            supported_gpus = ["A100", "A6000", "RTX 30", "RTX 40", "A30", "A40"]
            
            ampere_detected = False
            for gpu_name in gpu_names:
                if any(supported_gpu in gpu_name for supported_gpu in supported_gpus):
                    ampere_detected = True
                    break
            
            return ampere_detected
        except Exception as e:
            logging.warning(f"Error occurred while checking GPU: {e}")
            return False


    def load_model(self):
        if self.llm is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if is_ampere_gpu:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map='cuda',
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                warnings.warn(f"Your GPU is not Ampere! Disabling the use of flash_attention module")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map='cuda',
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=self.max_new_tokens)
            self.llm = HuggingFacePipeline(pipeline=pipe)
        return self.llm

    def create_retriever(self, vectorstore, search_type="similarity"):
        k = self.k_chunks
        if self.retriever is None:
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k})
        return self.retriever

    def create_retrievalqa_chain(self, retriever, prompt_template):
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        if self.retrievalQA is None:
            self.retrievalQA = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
        return self.retrievalQA

    def query_rag(self, query):
        if self.retrievalQA is None:
            raise ValueError("RetrievalQA chain has not been initialized.")
        return self.retrievalQA.invoke({"query": query})


    def create_index(self, directory=None, index_path='faiss_index'):
        if directory is None:
            directory = self.directory



        # Prepare documents
        docs = self.prepare_documents(directory)
        
        # Create vectorstore (index)
        self.create_vectorstore(docs)
        
        # Save the index to the specified path
        index_path = directory + '/' + index_path
        
        self.save_index(index_path)
        
        print(f"Index created and saved at {index_path}")

    def rag_pipeline(self, query, directory=None, prompt_template=None):
        if directory is None:
            directory = self.directory

        try:
            docs = self.prepare_documents(directory)
            vectorstore = self.create_vectorstore(docs)
            retriever = self.create_retriever(vectorstore)

            if prompt_template is None:
                prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
                1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
                2. If you find the answer, write the answer in a concise way with five sentences maximum.
                
                {context}
                
                Question: {question}
                
                Helpful Answer:
                """
                
            self.create_retrievalqa_chain(retriever, prompt_template)
            result = self.query_rag(query)
            
            print("Result:\n", result['result'])
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")


