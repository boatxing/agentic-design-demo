# pip install crewai langchain-openai

import os
from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 方式1: 使用自定义 LLM（例如 DeepSeek）
api_key = os.getenv("API_KEY")
if api_key:
    # 直接使用 LangChain ChatOpenAI 实例，CrewAI 会自动识别
    custom_llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2-251201",
        base_url="https://api.qnaigc.com/v1",
        api_key=api_key,
        temperature=0,
    )
    print(f"✅ 使用自定义 LLM: {custom_llm.model_name}")
else:
    raise ValueError(
        "未设置 API_KEY 环境变量。请设置 API_KEY 后再运行脚本。"
    )


@tool("股票价格查询工具")
def get_stock_price(ticker: str) -> float:
    """
    获取指定股票代码的最新模拟价格。返回float，未找到则抛出ValueError。
    """
    logging.info(f"工具调用：get_stock_price, 股票代码'{ticker}'")
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }
    price = simulated_prices.get(ticker.upper())

    if price is not None:
        return price
    else:
        raise ValueError(f"未找到'{ticker.upper()}' 的模拟价格。")


# 方式1: 在 Agent 级别设置 LLM（每个 Agent 可以使用不同的 LLM）
financial_analyst_agent = Agent(
    role="高级金融分析师",
    goal="使用工具分析股票数据并报告关键价格。",
    backstory="你是一名经验丰富的金融分析师，擅长使用数据源查找股票信息，回答简明直接。",
    verbose=True,
    tools=[get_stock_price],
    allow_delegation=False,
    llm=custom_llm,  # 为这个 Agent 指定自定义 LLM
)

analyze_aapl_task = Task(
    description=(
        "苹果（AAPL）当前模拟股价是多少？请用'股票价格查询工具'查找。"
        "如果未找到代码，需明确说明无法获取价格。"
    ),
    expected_output=(
        "用一句话说明AAPL 的模拟股价，如：'AAPL 的模拟股价为$178.15。'"
        "如果无法找到价格，也要明确说明。"
    ),
    agent=financial_analyst_agent,
)

# 方式2: 在 Crew 级别设置 LLM（会应用到所有没有单独设置 LLM 的 Agents）
financial_crew = Crew(
    agents=[financial_analyst_agent],
    tasks=[analyze_aapl_task],
    verbose=True,
    llm=custom_llm,  # 为整个 Crew 设置默认 LLM
)


def main():
    """主函数运行Crew。"""
    print("\n## 启动金融Crew...")
    print("----------------------")

    result = financial_crew.kickoff()

    print("\n----------------------")
    print("## Crew 执行结束。")
    print("\n最终结果:\n", result)


if __name__ == "__main__":
    main()
