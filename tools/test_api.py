from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-cDNrDh4dQfjiVnyby9B4K3NefSLvFhRlbbVMg3pqhKS1707p",
    base_url="https://api.a1r.cc/v1"
)

# 测试图片URL
image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

try:
    print("正在调用 API...")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "图片里面有什么？"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
            ]
        }],
        max_tokens=300,
        stream=False
    )
    
    # 调试：打印响应类型和内容
    print(f"\n响应类型: {type(response)}")
    print(f"响应内容: {response}")
    
    # 如果是字符串，直接打印
    if isinstance(response, str):
        print("\n⚠️  API 返回了字符串而不是对象")
        print(f"返回内容: {response}")
    else:
        # 正常处理响应
        print("\n" + "="*50)
        print("API 调用成功!")
        print("="*50)
        
        # 检查响应对象是否有 choices 属性
        if hasattr(response, 'choices') and len(response.choices) > 0:
            assistant_message = response.choices[0].message.content
            print(f"\n助手回复:\n{assistant_message}")
            
            # 打印使用情况
            if hasattr(response, 'usage'):
                print(f"\n" + "-"*50)
                print("Token 使用情况:")
                print(f"  - 提示 tokens: {response.usage.prompt_tokens}")
                print(f"  - 完成 tokens: {response.usage.completion_tokens}")
                print(f"  - 总计 tokens: {response.usage.total_tokens}")
            
            print(f"\n结束原因: {response.choices[0].finish_reason}")
        else:
            print("\n⚠️  响应对象没有 choices 属性")
            print(f"响应对象属性: {dir(response)}")
    
except Exception as e:
    print(f"\n❌ API 调用失败:")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    
    # 打印详细的堆栈跟踪
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()