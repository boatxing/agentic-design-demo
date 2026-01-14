import asyncio
import os
import nest_asyncio

from dotenv import load_dotenv
from langchain_classic.agents import (  # type: ignore[import-untyped]
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI

load_dotenv()

try:
    # 初始化具备工具调用能力的模型
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY 未在环境变量中找到")
    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2-251201",
        base_url="https://api.qnaigc.com/v1",
        api_key=api_key,  # type: ignore[arg-type]
        temperature=0,
    )
    print(f"语言模型初始化成功：{llm.model_name}")
except Exception as e:
    print(f":emoji: 初始化语言模型出错：{e}")
    llm = None

# ‑‑‑ 定义工具‑‑‑


@langchain_tool
def search_information(query: str) -> str:
    """
    根据主题提供事实信息。用于回答如"法国首都"或"伦敦天气？"等问题。
    """
    print(f"\n‑‑‑ :tools: ✅ 工具调用：search_information, 查询：'{query}' ‑‑‑")
    # 用预设结果模拟搜索工具
    simulated_results = {
        "weather in london": "伦敦当前天气多云，气温15°C。",
        "capital of france": "法国的首都是巴黎。",
        "population of earth": "地球人口约80 亿。",
        "tallest mountain": "珠穆朗玛峰是海􏆧最高的山峰。",
        "default": f"模拟搜索'{query}'：未找到具体信息，但该主题很有趣。",
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"‑‑‑ 工具结果：{result} ‑‑‑")
    return result


tools = [search_information]

# ‑‑‑ 创建工具调用Agent ‑‑‑
if llm:
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个乐于助人的助手。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

    async def run_agent_with_tool(query: str):
        """用Agent 执行查询并打印最终回复。"""
        print(f"\n‑‑‑ :emoji: Agent 运行查询：'{query}' ‑‑‑")
        try:
            response = await agent_executor.ainvoke({"input": query})
            print("\n‑‑‑ ✅ Agent 最终回复‑‑‑")
            print(response["output"])
        except Exception as e:
            print(f"\n:emoji: Agent 执行出错：{e}")

    async def main():
        """并发运行多个Agent 查询。"""
        tasks = [
            run_agent_with_tool("法国的首都是什么？"),
            run_agent_with_tool("伦敦天气如何？"),
            run_agent_with_tool("说说狗的相关信息。"),  # 触发默认工具回复
        ]
        await asyncio.gather(*tasks)

    nest_asyncio.apply()
    asyncio.run(main())
