import requests
import json

def test_ollama():
    """测试Ollama和llama3.1是否可用"""
    print("🔍 测试Ollama服务...")
    
    # 测试Ollama服务是否运行
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama服务正在运行")
            version_info = response.json()
            print(f"版本: {version_info.get('version', 'Unknown')}")
        else:
            print("❌ Ollama服务无响应")
            return False
    except Exception as e:
        print(f"❌ 无法连接Ollama服务: {e}")
        print("请确保运行: ollama serve")
        return False
    
    # 测试llama3.1模型
    print("\n🤖 测试llama3.1模型...")
    try:
        data = {
            "model": "qwen3:4b",
            "prompt": "你好，请回复一个字：好",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            reply = result.get("response", "").strip()
            print(f"✅ llama3.1模型工作正常")
            print(f"回复: {reply}")
            return True
        else:
            print(f"❌ llama3.1模型请求失败: {response.status_code}")
            print(f"错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ llama3.1模型测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama()
    if success:
        print("\n🎉 llama3.1可以正常使用！")
    else:
        print("\n❌ llama3.1不可用，请检查Ollama设置")