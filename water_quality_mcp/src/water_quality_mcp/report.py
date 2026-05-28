from openai import OpenAI

MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"


def generate_report(api_key, summary):
    """基于 GB 3838-2002 生成水环境溯源报告。"""
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"你是水环境溯源专家。请依据《城市黑臭水体整治工作指南》(2015)和GB 3838-2002撰写溯源报告。\n{summary}"}],
        temperature=0.1,
        max_tokens=2000
    )
    return resp.choices[0].message.content
