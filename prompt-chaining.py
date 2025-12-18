import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# 方式1：使用 ChatOpenAI 配置自定义 base_url
model = ChatOpenAI(
    model="deepseek-r1-0528",
    base_url="https://api.qnaigc.com/v1",
    api_key=os.getenv("API_KEY"),  # 或者使用 os.getenv("API_KEY")
    temperature=0
)

# ‑‑‑ 提示1：信息提取‑‑‑
prompt_extract = ChatPromptTemplate.from_template(
    "请从以下文本中提取技术规格：\n\n{text_input}"
)

# ‑‑‑ 提示2：转为JSON ‑‑‑
prompt_transform = ChatPromptTemplate.from_template(
    "请将以下技术规格转为JSON 格式，包含'cpu'、'memory' 和'storage' 三个键：\n\n{specifications}"
)

# ‑‑‑ 用LCEL 构建链‑‑‑
# StrOutputParser() 将LLM 消息输出转为字符串
extraction_chain = prompt_extract | model | StrOutputParser()

# 全链将提取链的输出作为'specifications' 变量传递给转换提示
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | model
    | StrOutputParser()
)

# ‑‑‑ 运行链‑‑‑
input_text = "新款笔记本配备3.5GHz 八核处理器、16GB 内存和1TB NVMe SSD。"

# 用输入文本字典执行链
final_result = full_chain.invoke({"text_input": input_text})

print("\n‑‑‑ 最终JSON 输出‑‑‑")
print(final_result)
