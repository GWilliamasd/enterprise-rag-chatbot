# 🤖 企业级RAG智能问答系统

基于 LangChain + Streamlit 构建的企业级检索增强生成（RAG）系统，支持多种文档格式和多个大语言模型。

## ✨ 特性

### 🧠 多模型支持
- **Ollama 本地模型**: llama3.1:latest, qwen3:4b
- **OpenAI API**: gpt-4o-mini
- **实时模型切换**: 无需重启，即时生效

### 📚 文档处理
- **多格式支持**: PDF, Word, Excel, TXT, CSV
- **智能分块**: 可配置文档切分策略
- **向量存储**: ChromaDB 持久化存储
- **批量上传**: 支持多文件同时处理

### 🔍 智能检索
- **语义搜索**: 基于向量相似度的文档检索
- **上下文构建**: 智能组合相关文档片段
- **来源追溯**: 显示答案的文档来源

### 🎨 用户界面
- **现代化UI**: 基于 Streamlit 的响应式界面
- **实时聊天**: 类ChatGPT的对话体验
- **数据可视化**: 响应时间、查询统计图表
- **配置面板**: 侧边栏实时参数调节

### 🚀 性能优化
- **智能缓存**: SQLite 查询结果缓存
- **异常处理**: 完善的错误恢复机制
- **系统监控**: 健康检查和状态监控

## 🛠️ 技术架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  EnterpriseRAG   │────│   Vector Store  │
│                 │    │     System       │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            ┌───────▼────┐ ┌────▼────┐ ┌───▼──────┐
            │   Ollama   │ │ OpenAI  │ │HuggingFace│
            │    LLM     │ │   API   │ │Embeddings │
            └────────────┘ └─────────┘ └──────────┘
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Ollama (可选，用于本地模型)
- OpenAI API Key (可选，用于GPT模型)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/ent-rag.git
cd ent-rag
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境** (可选)
```bash
# 创建 .env 文件
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. **安装 Ollama 模型** (可选)
```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull llama3.1
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

5. **运行应用**
```bash
streamlit run main.py
```

## 📋 使用说明

### 1. 系统初始化
- 打开浏览器访问 `http://localhost:8501`
- 点击 "🚀 初始化RAG系统"

### 2. 文档上传
- 切换到 "📁 文档管理" 标签
- 上传 PDF、Word、Excel 等文档
- 等待系统处理和向量化

### 3. 智能问答
- 在 "💬 智能问答" 标签中提问
- 系统会基于上传的文档回答问题
- 查看答案来源和相关文档

### 4. 模型切换
- 在侧边栏选择不同的语言模型
- 点击 "🔄 应用新配置" 即时切换
- 比较不同模型的回答效果

### 5. 系统监控
- "📊 数据分析" 查看使用统计
- "📋 系统日志" 查看操作记录

## ⚙️ 配置说明

### 模型配置
```python
# 在 main.py 中的 load_config()
{
    "llm_type": "ollama",  # 或 "openai"
    "llm_config": {
        "model": "llama3.1:latest",
        "temperature": 0.1
    },
    "embedding_type": "huggingface",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retrieval_k": 5
}
```

### 环境变量
```bash
OPENAI_API_KEY=your_openai_api_key
```

## 📁 项目结构

```
ent_RAG/
├── main.py              # Streamlit 主应用
├── rag_system.py        # RAG 系统核心
├── test_llama.py        # 模型测试工具
├── CLAUDE.md           # Claude Code 配置
├── requirements.txt     # Python 依赖
├── README.md           # 项目说明
├── .gitignore          # Git 忽略文件
└── .env                # 环境变量 (需创建)
```

## 🔧 开发指南

详见 [CLAUDE.md](./CLAUDE.md) 文件，包含：
- 开发命令和工具
- 代码架构说明
- 常见问题解决

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

⭐ 如果这个项目对你有帮助，请给个 Star！