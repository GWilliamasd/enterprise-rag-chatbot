import streamlit as st
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 导入企业RAG系统（假设在rag_system.py文件中）
try:
    from rag_system import EnterpriseRAGSystem, DocumentProcessor
except ImportError:
    st.error("请确保 rag_system.py 文件存在并包含 EnterpriseRAGSystem 类")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="🤖 企业级RAG智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .bot-message {
        background-color: #e8f4fd;
    }
    .source-doc {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .upload-zone {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


# 初始化会话状态
def init_session_state():
    """初始化Streamlit会话状态"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = 0
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = []


def load_config():
    """加载RAG系统配置"""
    return {
        "llm_type": "ollama",  # 使用Ollama
        "llm_config": {
            "model": "llama3.1:latest",
            "temperature": 0.1,
            "base_url": "http://localhost:11434"
        },
        "embedding_type": "huggingface",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_k": 5
    }


def initialize_rag_system(sidebar_config=None):
    """初始化RAG系统"""
    if not st.session_state.system_initialized:
        with st.spinner("🚀 正在初始化RAG系统..."):
            try:
                config = load_config()
                
                # 使用侧边栏配置更新系统配置
                if sidebar_config:
                    config["llm_config"]["model"] = sidebar_config["llm_model"]
                    config["llm_config"]["temperature"] = sidebar_config["temperature"]
                    config["retrieval_k"] = sidebar_config["retrieval_k"]
                    config["embedding_model"] = sidebar_config["embedding_model"]
                    
                    # 根据LLM模型设置类型
                    if sidebar_config["llm_model"] == "gpt-4o-mini":
                        config["llm_type"] = "openai"
                        config["llm_config"]["api_key"] = os.getenv("OPENAI_API_KEY")
                    else:
                        config["llm_type"] = "ollama"
                    
                    # 根据embedding模型设置类型
                    if sidebar_config["embedding_model"].startswith("sentence-transformers"):
                        config["embedding_type"] = "huggingface"
                    else:
                        config["embedding_type"] = "ollama"
                
                st.session_state.rag_system = EnterpriseRAGSystem(config)
                st.session_state.system_initialized = True
                st.session_state.current_config = sidebar_config  # 保存当前配置
                st.success("✅ RAG系统初始化成功！")
                return True
            except Exception as e:
                st.error(f"❌ 系统初始化失败: {str(e)}")
                return False
    return True


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("🛠️ 系统控制台")

    # 系统状态
    st.sidebar.subheader("📊 系统状态")
    if st.session_state.system_initialized:
        st.sidebar.success("🟢 系统已就绪")
    else:
        st.sidebar.error("🔴 系统未初始化")

    # 统计信息
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📄 已处理文档", st.session_state.documents_processed)
    with col2:
        st.metric("💬 对话次数", len(st.session_state.chat_history))

    # 配置选项
    st.sidebar.subheader("⚙️ 系统配置")

    # LLM模型选择
    available_models = ["llama3.1:latest", "qwen3:4b"]
    
    # 检查是否有OpenAI API密钥
    if os.getenv("OPENAI_API_KEY"):
        available_models.append("gpt-4o-mini")
    
    llm_model = st.sidebar.selectbox(
        "选择语言模型",
        available_models,
        index=0,
        help="用于生成回答的大语言模型"
    )
    
    # Embedding模型选择
    embedding_model = st.sidebar.selectbox(
        "选择嵌入模型",
        ["sentence-transformers/all-MiniLM-L6-v2", "nomic-embed-text"],
        index=0,
        help="用于文本向量化的模型"
    )

    # 检索参数
    retrieval_k = st.sidebar.slider("检索文档数量", 1, 10, 5)
    temperature = st.sidebar.slider("模型温度", 0.0, 1.0, 0.1, 0.1)

    # 高级选项
    with st.sidebar.expander("🔧 高级选项"):
        chunk_size = st.sidebar.slider("文档块大小", 500, 2000, 1000, 100)
        chunk_overlap = st.sidebar.slider("文档块重叠", 50, 500, 200, 50)
        enable_cache = st.sidebar.checkbox("启用缓存", True)

    # 系统重置
    if st.sidebar.button("🔄 重置系统", type="secondary"):
        st.session_state.clear()
        st.rerun()

    return {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "retrieval_k": retrieval_k,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "enable_cache": enable_cache
    }


def render_document_upload():
    """渲染文档上传界面"""
    st.subheader("📁 文档管理")

    # 文档上传区域
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown("### 📤 上传文档到知识库")
    st.markdown("支持格式：PDF、Word、Excel、TXT")

    uploaded_files = st.file_uploader(
        "选择文件",
        type=['pdf', 'docx', 'xlsx', 'txt'],
        accept_multiple_files=True,
        key="doc_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 集合名称输入
    collection_name = st.text_input(
        "知识库名称",
        value="default",
        help="为您的文档集合命名"
    )

    # 处理上传的文件
    if uploaded_files and st.button("🚀 开始处理文档", type="primary"):
        process_uploaded_documents(uploaded_files, collection_name)

    # 显示已上传文档列表
    render_document_list()


def process_uploaded_documents(uploaded_files: List, collection_name: str):
    """处理上传的文档"""
    if not st.session_state.system_initialized:
        st.error("❌ 请先初始化系统")
        return

    # 创建临时目录保存上传的文件
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = []

    # 保存上传的文件
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    # 显示处理进度
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 调用RAG系统处理文档
        status_text.text("📝 正在处理文档...")
        result = st.session_state.rag_system.ingest_documents(file_paths, collection_name)

        progress_bar.progress(100)

        if result['status'] == 'success':
            st.success(f"✅ 成功处理 {result['documents_processed']} 个文档块")
            st.session_state.documents_processed += len(uploaded_files)

            # 记录处理日志
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "文档上传",
                "files": [f.name for f in uploaded_files],
                "collection": collection_name,
                "status": "成功"
            }
            st.session_state.processing_logs.append(log_entry)

        else:
            st.error(f"❌ 处理失败: {result['message']}")

    except Exception as e:
        st.error(f"❌ 处理过程中出错: {str(e)}")

    finally:
        # 清理临时文件
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass

        progress_bar.empty()
        status_text.empty()


def render_document_list():
    """显示已处理文档列表"""
    if st.session_state.processing_logs:
        st.subheader("📋 处理历史")

        # 转换为DataFrame显示
        df = pd.DataFrame(st.session_state.processing_logs)

        # 只显示最近的10条记录
        recent_logs = df.tail(10).iloc[::-1]  # 逆序显示

        for _, log in recent_logs.iterrows():
            with st.expander(f"📄 {log['timestamp']} - {log['action']}"):
                st.write(f"**文件:** {', '.join(log['files'])}")
                st.write(f"**集合:** {log['collection']}")
                st.write(f"**状态:** {log['status']}")


def render_chat_interface():
    """渲染聊天界面"""
    st.subheader("💬 智能问答")

    if not st.session_state.system_initialized:
        st.warning("⚠️ 请先初始化系统才能开始对话")
        return

    # 显示聊天历史
    chat_container = st.container()

    with chat_container:
        for i, (question, answer, sources, response_time) in enumerate(st.session_state.chat_history):
            # 用户消息
            st.markdown(f'''
            <div class="chat-message user-message">
                <strong>🙋 您:</strong> {question}
            </div>
            ''', unsafe_allow_html=True)

            # AI回答
            st.markdown(f'''
            <div class="chat-message bot-message">
                <strong>🤖 AI助手:</strong> {answer}
                <br><small>⏱️ 响应时间: {response_time:.2f}秒</small>
            </div>
            ''', unsafe_allow_html=True)

            # 显示来源文档
            if sources:
                with st.expander(f"📚 查看来源文档 ({len(sources)}个)"):
                    for j, source in enumerate(sources):
                        st.markdown(f'''
                        <div class="source-doc">
                            <strong>来源 {j + 1}:</strong> {source['metadata'].get('source', '未知来源')}<br>
                            <em>{source['content'][:200]}...</em>
                        </div>
                        ''', unsafe_allow_html=True)

    # 聊天输入框
    st.markdown("---")

    # 使用form来处理输入
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input(
                "请输入您的问题",
                placeholder="例如：公司的休假政策是什么？",
                key="user_question"
            )

        with col2:
            submit_button = st.form_submit_button("发送 📤", type="primary")

    # 处理用户输入
    if submit_button and user_input:
        handle_user_query(user_input)


def handle_user_query(question: str):
    """处理用户查询"""
    if not st.session_state.rag_system:
        st.error("❌ 系统未初始化")
        return

    with st.spinner("🤔 AI正在思考中..."):
        try:
            # 调用RAG系统查询
            start_time = time.time()
            response = st.session_state.rag_system.query(question, user_id="streamlit_user")
            end_time = time.time()

            # 添加到聊天历史
            st.session_state.chat_history.append((
                question,
                response['answer'],
                response.get('source_documents', []),
                response.get('response_time', end_time - start_time)
            ))

            # 刷新页面显示新消息
            st.rerun()

        except Exception as e:
            st.error(f"❌ 查询失败: {str(e)}")


def render_analytics_dashboard():
    """渲染分析仪表板"""
    st.subheader("📊 系统分析")

    if not st.session_state.system_initialized:
        st.info("ℹ️ 初始化系统后可查看分析数据")
        return

    # 获取系统统计
    try:
        stats = st.session_state.rag_system.get_system_stats()

        # 显示关键指标
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('''
            <div class="metric-card">
                <h3>📄</h3>
                <h2>{}</h2>
                <p>向量库大小</p>
            </div>
            '''.format(stats.get('vector_store_size', 0)), unsafe_allow_html=True)

        with col2:
            st.markdown('''
            <div class="metric-card">
                <h3>💬</h3>
                <h2>{}</h2>
                <p>总查询次数</p>
            </div>
            '''.format(stats.get('total_queries', 0)), unsafe_allow_html=True)

        with col3:
            st.markdown('''
            <div class="metric-card">
                <h3>⚡</h3>
                <h2>{:.2f}s</h2>
                <p>平均响应时间</p>
            </div>
            '''.format(stats.get('avg_response_time', 0)), unsafe_allow_html=True)

        with col4:
            st.markdown('''
            <div class="metric-card">
                <h3>🎯</h3>
                <h2>{}</h2>
                <p>今日查询</p>
            </div>
            '''.format(len(st.session_state.chat_history)), unsafe_allow_html=True)

        # 响应时间趋势图
        if st.session_state.chat_history:
            st.subheader("📈 响应时间趋势")

            response_times = [chat[3] for chat in st.session_state.chat_history]
            query_numbers = list(range(1, len(response_times) + 1))

            fig = px.line(
                x=query_numbers,
                y=response_times,
                title="查询响应时间变化",
                labels={'x': '查询次数', 'y': '响应时间 (秒)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # 查询长度分布
        if st.session_state.chat_history:
            st.subheader("📊 查询长度分布")

            query_lengths = [len(chat[0]) for chat in st.session_state.chat_history]

            fig = px.histogram(
                x=query_lengths,
                nbins=10,
                title="用户查询长度分布",
                labels={'x': '查询长度 (字符)', 'y': '频次'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 获取统计信息失败: {str(e)}")


def render_system_logs():
    """渲染系统日志"""
    st.subheader("📋 系统日志")

    # 查看处理日志
    if st.session_state.processing_logs:
        st.write("**文档处理日志:**")
        df = pd.DataFrame(st.session_state.processing_logs)
        st.dataframe(df, use_container_width=True)

    # 查看查询日志
    if st.session_state.chat_history:
        st.write("**查询历史:**")
        chat_df = pd.DataFrame([
            {
                "时间": datetime.now().strftime("%H:%M:%S"),
                "问题": chat[0][:50] + "..." if len(chat[0]) > 50 else chat[0],
                "响应时间": f"{chat[3]:.2f}s",
                "来源数量": len(chat[2])
            }
            for chat in st.session_state.chat_history
        ])
        st.dataframe(chat_df, use_container_width=True)

    # 导出日志
    if st.button("📥 导出日志"):
        export_logs()


def export_logs():
    """导出系统日志"""
    try:
        log_data = {
            "processing_logs": st.session_state.processing_logs,
            "chat_history": [
                {
                    "question": chat[0],
                    "answer": chat[1],
                    "response_time": chat[3],
                    "source_count": len(chat[2])
                }
                for chat in st.session_state.chat_history
            ],
            "export_time": datetime.now().isoformat()
        }

        log_json = json.dumps(log_data, ensure_ascii=False, indent=2)

        st.download_button(
            label="📥 下载日志文件",
            data=log_json,
            file_name=f"rag_system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ 导出失败: {str(e)}")


def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()

    # 主标题
    st.markdown('<h1 class="main-header">🤖 企业级RAG智能问答系统</h1>', unsafe_allow_html=True)

    # 渲染侧边栏
    sidebar_config = render_sidebar()

    # 主要内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["💬 智能问答", "📁 文档管理", "📊 数据分析", "📋 系统日志"])

    # 初始化系统
    if not st.session_state.system_initialized:
        if st.button("🚀 初始化RAG系统", type="primary", key="init_system"):
            initialize_rag_system(sidebar_config)
    else:
        # 系统已初始化，显示当前使用的模型
        if hasattr(st.session_state, 'current_config'):
            current_model = st.session_state.current_config.get('llm_model', '未知')
            st.sidebar.success(f"🤖 当前模型: {current_model}")
        
        # 检查配置是否有变化
        if st.session_state.get('current_config') != sidebar_config:
            if st.sidebar.button("🔄 应用新配置", type="primary"):
                with st.spinner("正在更新系统配置..."):
                    try:
                        # 先清空缓存以确保使用新模型
                        st.session_state.rag_system.clear_cache()
                        
                        # 更新配置
                        config = load_config()
                        config["llm_config"]["model"] = sidebar_config["llm_model"]
                        config["llm_config"]["temperature"] = sidebar_config["temperature"]
                        config["retrieval_k"] = sidebar_config["retrieval_k"]
                        config["embedding_model"] = sidebar_config["embedding_model"]
                        
                        # 根据LLM模型设置类型
                        if sidebar_config["llm_model"] == "gpt-4o-mini":
                            config["llm_type"] = "openai"
                            config["llm_config"]["api_key"] = os.getenv("OPENAI_API_KEY")
                        else:
                            config["llm_type"] = "ollama"
                        
                        # 根据embedding模型设置类型
                        if sidebar_config["embedding_model"].startswith("sentence-transformers"):
                            config["embedding_type"] = "huggingface"
                        else:
                            config["embedding_type"] = "ollama"
                        
                        # 更新RAG系统配置
                        success = st.session_state.rag_system.update_config(config)
                        if success:
                            st.session_state.current_config = sidebar_config
                            st.success(f"✅ 已切换到: {sidebar_config['llm_model']}")
                        else:
                            st.error("❌ 配置更新失败，请查看日志")
                    except Exception as e:
                        st.error(f"❌ 配置更新失败: {str(e)}")
            else:
                st.sidebar.warning("⚠️ 配置已变更，点击'应用新配置'生效")

    with tab1:
        render_chat_interface()

    with tab2:
        render_document_upload()

    with tab3:
        render_analytics_dashboard()

    with tab4:
        render_system_logs()

    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🔧 企业级RAG系统 | 基于LangChain + Streamlit构建"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
