import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ‑‑‑ 配置‑‑‑
# 从.env 文件加载环境变量（用于OPENAI_API_KEY）
load_dotenv()

# 检查API key 是否设置
if not os.getenv("API_KEY"):
    raise ValueError("API_KEY 未在.env 文件中找到，请添加。")

# 初始化Chat LLM，使用gpt‑4o，低温度保证输出确定性
llm = ChatOpenAI(
    model="deepseek/deepseek-v3.2-251201",
    base_url="https://api.qnaigc.com/v1",
    api_key=os.getenv("API_KEY"),  # 或者使用 os.getenv("API_KEY")
    temperature=0,
)
print(f"语言模型初始化成功：{llm.model_name}")


def run_reflection_loop():
    """
    演示多步AI 反思循环，逐步优化Python 函数。
    """
    # ‑‑‑ 核心任务‑‑‑
    task_prompt = """
    你的任务是创建一个名为`calculate_factorial` 的Python 函数。
    该函数需满足以下要求：
    第4 章：反思（Reflection）
    1. 只接受一个整数参数n。
    2. 计算其阶乘（n!）。
    3. 包含清晰的docstring，说明函数功能。
    4. 处理边界情况：0 的阶乘为1。
    5. 处理无效输入：若输入为负数则抛出ValueError。
    """
    # ‑‑‑ 反思循环‑‑‑
    max_iterations = 3
    current_code = ""
    # 构建对话历史，为每步提供上下文
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print("\n" + "=" * 25 + f" 反思循环：第{i + 1} 次迭代" + "=" * 25)

        # ‑‑‑ 1. 生成/优化阶段‑‑‑
        # 首次迭代为生成，后续为优化
        if i == 0:
            print("\n>>> 阶段1：生成初始代码...")
            # 首条消息为任务提示
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> 阶段1：根据批判优化代码...")
            # 消息历史包含任务、上次代码和批判
            # 指示模型应用批判意见优化代码
            message_history.append(HumanMessage(content="请根据批判意见优化代码。"))
            response = llm.invoke(message_history)
            current_code = response.content

        print("\n‑‑‑ 生成代码（第" + str(i + 1) + " 版）‑‑‑\n" + current_code)
        message_history.append(response)  # 将生成代码加入历史

        # ‑‑‑ 2. 反思阶段‑‑‑
        print("\n>>> 阶段2：对生成代码进行反思...")

        # 为批判者Agent 创建专用提示
        # 要求模型以资深代码审查员身份批判代码
        reflector_prompt = [
            SystemMessage(
                content="""
                你是一名资深软件工程师，精通Python。
                你的职责是对提供的Python 代码进行细致代码审查。
                请根据原始任务要求，严格评估代码。
                检查是否有bug、风格问题、遗漏边界情况及其他可改进之处。
                若代码完美且满足所有要求，仅回复'CODE_IS_PERFECT'。
                否则，请以项目符号列表形式给出批判意见。
                """
            ),
            HumanMessage(
                content=f"原始任务：\n{task_prompt}\n\n 待审查代码：\n{current_code}"
            ),
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # ‑‑‑ 3. 停止条件‑‑‑
        if "CODE_IS_PERFECT" in critique:
            print("\n‑‑‑ 批判‑‑‑\n 未发现进一步批判，代码已达要求。")
            break

        print("\n‑‑‑ 批判‑‑‑\n" + critique)
        # 将批判意见加入历史，供下轮优化使用
        message_history.append(HumanMessage(content=f"上次代码批判意见：\n{critique}"))

    print("\n" + "=" * 30 + " 最终结果" + "=" * 30)
    print("\n 反思流程优化后的最终代码：\n")
    print(current_code)


if __name__ == "__main__":
    run_reflection_loop()
