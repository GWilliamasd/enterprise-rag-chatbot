"""
企业级RAG系统实现
基于langchain构建的完整RAG解决方案
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3

from curl_cffi import Response
from dotenv import load_dotenv
load_dotenv()
# langchain imports
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

class SmartCacheSystem:
    """智能缓存系统"""
    
    def __init__(self, cache_file: str = "rag_cache.db"):
        self.cache_file = cache_file
        self._init_database()
    
    def _init_database(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """获取缓存的响应"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT response FROM query_cache WHERE query_hash = ?',
            (query_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def cache_response(self, query: str, response: str):
        """缓存响应"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO query_cache (query_hash, query_text, response) VALUES (?, ?, ?)',
            (query_hash, query, response)
        )
        conn.commit()
        conn.close()

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".csv": CSVLoader
        }
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.loaders:
            raise ValueError(f"不支持的文件类型: {file_ext}")
        
        try:
            loader_class = self.loaders[file_ext]
            loader = loader_class(file_path)
            documents = loader.load()
            
            # 添加文件路径到元数据
            for doc in documents:
                doc.metadata['source'] = file_path
                doc.metadata['file_type'] = file_ext
                doc.metadata['processed_time'] = datetime.now().isoformat()
            
            return documents
        
        except Exception as e:
            logging.error(f"加载文档失败 {file_path}: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"文档分割失败: {str(e)}")
            return documents

class EnterpriseRAGSystem:
    """企业级RAG系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.cache_system = SmartCacheSystem()
        self.doc_processor = DocumentProcessor()
        
        # 初始化组件
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 初始化LLM
            self._initialize_llm()
            
            # 初始化嵌入模型
            self._initialize_embeddings()
            
            self.logger.info("RAG系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {str(e)}")
            raise
    
    def _initialize_llm(self):
        """初始化语言模型"""
        llm_config = self.config.get("llm_config", {})
        
        if self.config.get("llm_type") == "ollama":
            self.llm = OllamaLLM(
                model=llm_config.get("model", "llama3.1:latest"),
                temperature=llm_config.get("temperature", 0.1),
                base_url=llm_config.get("base_url", "http://localhost:11434")
            )
        elif self.config.get("llm_type") == "openai":
            self.llm = ChatOpenAI(
                model=llm_config.get("model", "gpt-4o-mini"),
                temperature=llm_config.get("temperature", 0.1),
                api_key=llm_config.get("api_key")
            )
        else:
            raise ValueError(f"不支持的LLM类型: {self.config.get('llm_type')}")
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        if self.config.get("embedding_type") == "huggingface":
            model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        elif self.config.get("embedding_type") == "ollama":
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            # 默认使用HuggingFace
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def ingest_documents(self, file_paths: List[str], collection_name: str = "default") -> Dict[str, Any]:
        """批量导入文档"""
        self.logger.info(f"开始导入{len(file_paths)}个文档到集合: {collection_name}")
        
        try:
            # 加载所有文档
            all_docs = []
            processed_files = []
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    docs = self.doc_processor.load_document(file_path)
                    if docs:
                        all_docs.extend(docs)
                        processed_files.append(file_path)
                        self.logger.info(f"成功加载文档: {file_path}")
                    else:
                        self.logger.warning(f"文档加载为空: {file_path}")
                else:
                    self.logger.warning(f"文件不存在: {file_path}")
            
            if not all_docs:
                return {"status": "error", "message": "没有成功加载任何文档"}
            
            # 分割文档
            split_docs = self.doc_processor.split_documents(all_docs)
            self.logger.info(f"文档分割完成，共{len(split_docs)}个块")
            
            # 创建或更新向量存储
            vector_db_path = f"./vectordb/{collection_name}"
            os.makedirs(vector_db_path, exist_ok=True)
            
            if self.vector_store is None:
                # 创建新的向量存储
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    persist_directory=vector_db_path,
                    collection_name=collection_name
                )
            else:
                # 添加到现有向量存储
                self.vector_store.add_documents(split_docs)
            
            # 持久化存储
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
            
            # 创建检索器
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.get("retrieval_k", 5)}
            )
            
            self.logger.info(f"成功导入{len(split_docs)}个文档块")
            
            return {
                "status": "success",
                "documents_processed": len(split_docs),
                "files_processed": processed_files,
                "collection_name": collection_name
            }
            
        except Exception as e:
            error_msg = f"文档导入失败: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def query(self, question: str, user_id: str = None) -> Dict[str, Any]:
        """查询RAG系统"""
        self.logger.info(f"收到查询: {question}")
        
        try:
            start_time = datetime.now()
            
            # 检查缓存（但为了测试模型切换，暂时禁用）
            # cached_response = self.cache_system.get_cached_response(question)
            # if cached_response:
            #     self.logger.info("从缓存返回响应")
            #     return {
            #         "answer": cached_response,
            #         "source_documents": [],
            #         "response_time": 0.1,
            #         "from_cache": True
            #     }
            
            # 检查系统是否就绪
            if self.retriever is None:
                return {
                    "answer": "系统尚未加载任何文档，请先上传文档到知识库。",
                    "source_documents": [],
                    "response_time": 0,
                    "error": "no_documents"
                }
            
            # 检索相关文档
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。",
                    "source_documents": [],
                    "response_time": (datetime.now() - start_time).total_seconds()
                }
            
            # 构建上下文
            context = self._build_context(relevant_docs)
            
            # 生成答案
            answer = self._generate_answer(question, context)
            
            # 计算响应时间
            response_time = (datetime.now() - start_time).total_seconds()
            
            # 缓存响应
            self.cache_system.cache_response(question, answer)
            
            # 记录查询日志
            self._log_query(user_id, question, answer, response_time)
            
            # 格式化源文档
            formatted_sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
            
            return {
                "answer": answer,
                "source_documents": formatted_sources,
                "response_time": response_time,
                "from_cache": False
            }
            
        except Exception as e:
            error_msg = f"查询处理失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "answer": "抱歉，处理您的查询时出现错误，请稍后重试。",
                "source_documents": [],
                "response_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            }
    
    def _build_context(self, documents: List[Document]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            # 提取文件名
            if "/" in source or "\\" in source:
                source = os.path.basename(source)
            
            context_part = f"[文档{i}] 来源: {source}\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """生成答案"""
        prompt_template = """
你是一个专业的AI助手，基于提供的上下文信息回答用户问题。

上下文信息：
{context}

用户问题：{question}

回答要求：
1. 基于上下文信息提供准确、详细的回答
2. 如果上下文中没有足够信息回答问题，请明确说明
3. 保持回答的专业性和友好性
4. 可以适当引用具体的文档来源
5. 回答要条理清楚，易于理解

回答：
"""
        
        try:
            prompt = prompt_template.format(context=context, question=question)
            
            # 根据LLM类型调用不同方法
            if self.config.get("llm_type") == "openai":
                response = self.llm.invoke(prompt)
                return response.content.strip()
            else:
                # Ollama LLM
                response = self.llm(prompt)
                return response.strip()
        
        except Exception as e:
            self.logger.error(f"答案生成失败: {str(e)}")
            return "抱歉，生成回答时出现错误。"
    
    def _log_query(self, user_id: str, question: str, answer: str, response_time: float):
        """记录查询日志"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id or "anonymous",
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "response_time": response_time
            }
            
            # 写入日志文件
            with open("query_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            self.logger.error(f"日志记录失败: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "vector_store_size": 0,
            "total_queries": 0,
            "avg_response_time": 0,
            "cache_size": 0
        }
        
        try:
            # 向量存储大小
            if self.vector_store:
                try:
                    if hasattr(self.vector_store, '_collection'):
                        stats["vector_store_size"] = self.vector_store._collection.count()
                    else:
                        stats["vector_store_size"] = "未知"
                except:
                    stats["vector_store_size"] = "未知"
            
            # 查询统计
            try:
                if os.path.exists("query_logs.jsonl"):
                    with open("query_logs.jsonl", "r", encoding="utf-8") as f:
                        logs = [json.loads(line) for line in f if line.strip()]
                        stats["total_queries"] = len(logs)
                        
                        if logs:
                            response_times = [log.get("response_time", 0) for log in logs]
                            stats["avg_response_time"] = sum(response_times) / len(response_times)
            except Exception as e:
                self.logger.warning(f"读取查询日志失败: {str(e)}")
            
            # 缓存统计
            try:
                conn = sqlite3.connect(self.cache_system.cache_file)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM query_cache')
                stats["cache_size"] = cursor.fetchone()[0]
                conn.close()
            except Exception as e:
                self.logger.warning(f"读取缓存统计失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"获取系统统计失败: {str(e)}")
        
        return stats
    
    def load_existing_collection(self, collection_name: str = "default") -> bool:
        """加载已存在的文档集合"""
        try:
            vector_db_path = f"./vectordb/{collection_name}"
            
            if os.path.exists(vector_db_path):
                self.vector_store = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.get("retrieval_k", 5)}
                )
                
                self.logger.info(f"成功加载已存在的集合: {collection_name}")
                return True
            else:
                self.logger.info(f"集合不存在: {collection_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"加载集合失败: {str(e)}")
            return False
    
    def list_collections(self) -> List[str]:
        """列出所有可用的文档集合"""
        try:
            vectordb_dir = "./vectordb"
            if os.path.exists(vectordb_dir):
                collections = [
                    name for name in os.listdir(vectordb_dir)
                    if os.path.isdir(os.path.join(vectordb_dir, name))
                ]
                return collections
            return []
        except Exception as e:
            self.logger.error(f"列出集合失败: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除文档集合"""
        try:
            import shutil
            vector_db_path = f"./vectordb/{collection_name}"
            
            if os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)
                self.logger.info(f"成功删除集合: {collection_name}")
                
                # 如果删除的是当前集合，重置系统状态
                if self.vector_store and hasattr(self.vector_store, '_persist_directory'):
                    if self.vector_store._persist_directory == vector_db_path:
                        self.vector_store = None
                        self.retriever = None
                
                return True
            else:
                self.logger.warning(f"集合不存在: {collection_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除集合失败: {str(e)}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新系统配置"""
        try:
            self.config.update(new_config)
            
            # 重新初始化相关组件
            if "llm_config" in new_config or "llm_type" in new_config:
                self._initialize_llm()
            
            if "embedding_model" in new_config or "embedding_type" in new_config:
                self._initialize_embeddings()
                
            if "retrieval_k" in new_config and self.retriever:
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": new_config["retrieval_k"]}
                )
            
            self.logger.info("配置更新成功")
            return True
            
        except Exception as e:
            self.logger.error(f"配置更新失败: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 检查LLM
            health_status["components"]["llm"] = {
                "status": "ok" if self.llm else "error",
                "type": self.config.get("llm_type", "unknown")
            }
            
            # 检查嵌入模型
            health_status["components"]["embeddings"] = {
                "status": "ok" if self.embeddings else "error",
                "type": self.config.get("embedding_type", "unknown")
            }
            
            # 检查向量存储
            health_status["components"]["vector_store"] = {
                "status": "ok" if self.vector_store else "not_loaded",
                "has_documents": bool(self.retriever)
            }
            
            # 检查缓存系统
            try:
                conn = sqlite3.connect(self.cache_system.cache_file)
                conn.close()
                health_status["components"]["cache"] = {"status": "ok"}
            except:
                health_status["components"]["cache"] = {"status": "error"}
            
            # 检查日志系统
            health_status["components"]["logging"] = {
                "status": "ok" if os.path.exists("rag_system.log") else "warning"
            }
            
            # 总体状态评估
            component_statuses = [comp["status"] for comp in health_status["components"].values()]
            if "error" in component_statuses:
                health_status["status"] = "degraded"
            elif "warning" in component_statuses or "not_loaded" in component_statuses:
                health_status["status"] = "warning"
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    def clear_cache(self) -> bool:
        """清理缓存"""
        try:
            conn = sqlite3.connect(self.cache_system.cache_file)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM query_cache')
            conn.commit()
            conn.close()
            
            self.logger.info("缓存清理完成")
            return True
            
        except Exception as e:
            self.logger.error(f"缓存清理失败: {str(e)}")
            return False

# 工具函数
def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        "llm_type": "ollama",
        "llm_config": {
            "model": "llama3.1:latest",
            "temperature": 0.1
        },
        "embedding_type": "huggingface",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_k": 5,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

def test_system():
    """测试系统功能"""
    print("🧪 开始RAG系统测试...")
    
    # 创建配置
    config = create_default_config()
    
    try:
        # 初始化系统
        print("📊 初始化RAG系统...")
        rag_system = EnterpriseRAGSystem(config)
        
        # 健康检查
        print("🔍 执行健康检查...")
        health = rag_system.health_check()
        print(f"系统状态: {health['status']}")
        
        # 测试查询（无文档）
        print("❓ 测试空查询...")
        response = rag_system.query("测试问题")
        print(f"响应: {response['answer'][:100]}...")
        
        print("✅ 系统测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 系统测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_system()