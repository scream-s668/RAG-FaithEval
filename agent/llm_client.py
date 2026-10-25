from openai import OpenAI

API_KEY="sk-dummy-key-for-ci"
HAS_REAL_API_KEY = False

try:
    from agent.secret_key import REAL_API_KEY
    API_KEY=REAL_API_KEY
    HAS_REAL_API_KEY = True
except ImportError:
    print("⚠️ 未找到本地密钥文件，使用假 Key（适用于 CI 环境）")

BASE_URL= "https://api.deepseek.com"
MODEL_NAME = "deepseek-flash"

client=OpenAI(api_key=API_KEY,base_url=BASE_URL)

def ask_llm(prompt:str)->str:
    if API_KEY == "sk-dummy-key-for-ci":
        return "调用失败（无 Key）"

    try:
        response=client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role':'user','content':prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f'大模型调用失败：{e}')
        return "调用失败"