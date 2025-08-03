import os
import faiss
from faiss import IndexHNSWFlat
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.indices.vector_store.retrievers import (
            VectorIndexRetriever,
        )
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings
from openai import OpenAI, Client
from typing import List
from time import time

Embedding = List[float]

class LLM(BaseEmbedding):
    """Qwen3-Embedding."""
    model_name: str = "Qwen3-Embedding"
    embed_batch_size: int = 128
    system_prompt: str = 'You are a helpful assistant.'
    embed_system_prompt: str = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}'
    client: Client = OpenAI(
        api_key=os.getenv("API_KEY"), # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    dim: int = 1024

    def set_system_prompt(self, prompt: str) ->None:
        self.system_prompt = prompt
    
    def query(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model="qwen-plus",  # qwen-plus 属于 qwen3 模型，如需开启思考模式，请参见：https://help.aliyun.com/zh/model-studio/deep-thinking
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        )
        return completion.choices[0].message.content
    
    def emb_text(self, text: str) -> Embedding:
        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=self.dim,
            encoding_format="float"
        )
        return completion.data[0].embedding
    
    def emb_query(self, query: str) -> Embedding:
        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=self.embed_system_prompt.format(query=query),
            dimensions=self.dim,
            encoding_format="float"
        )
        return completion.data[0].embedding
    
    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """
        return self.emb_query(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query asynchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """
        return self.emb_query(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.

        Subclasses should implement this method. Reference get_text_embedding's
        docstring for more information.
        """
        return self.emb_text(text)

class RAG:
    def __init__(self, vector_db_path='vector_db/db-001', from_disk=False):
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
        self.vector_db_path = vector_db_path
        
        self.llm = LLM()
        Settings.embed_model = self.llm
        if not from_disk:
            self.faiss_index = IndexHNSWFlat(self.llm.dim, 2)
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = None
        else:
            self.vector_store = FaissVectorStore.from_persist_dir(self.vector_db_path)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=self.vector_db_path
            )
            self.index = load_index_from_storage(storage_context=self.storage_context)
        self.query_engine=None

    def build_db(self, docs: List[str]):
        docs = [Document(doc_id=str(id), text=doc) for id, doc in enumerate(docs) if doc!=""]
        self.index = VectorStoreIndex.from_documents(
            documents=docs, 
            storage_context=self.storage_context, 
            embed_model=self.llm,
            show_progress=True
        )
        print("vector db successfully built !")

    def save_db(self):
        self.index.storage_context.persist(self.vector_db_path)

    def buid_query_engine(self, similarity_top_k):
        self.query_engine = self.index.as_retriever(similarity_top_k=similarity_top_k,
                                                    embed_model=self.llm)
        print("query engine successfully built!")
    
    def generate(self, query: str, docs: List[str]) -> str:
        prompt = f"{query}\nRelated documents are:"
        for doc in docs:
            prompt += doc+"\n"
        return self.llm.query(prompt)

    def query(self, query: str) -> str:
        # start = time()
        node_scores = self.query_engine.retrieve(query)
        # end = time()
        # print(f"embed+retrieval: {end-start}s")
        docs = [node_s.node.text for node_s in node_scores]
        ans = self.generate(query, docs)
        return ans
        