from google.adk.agents import LlmAgent, SequentialAgent

# 第一个Agent 生成初稿
generator = LlmAgent(
    name="DraftWriter",
    description="根据主题生成初稿内容。",
    instruction="写一段简短、信息丰富的主题段落。",
    output_key="draft_text",  # 输出保存到此状态键
)

# 第二个Agent 批判初稿
reviewer = LlmAgent(
    name="FactChecker",
    description="审查文本的事实准确性并给出结构化批判。",
    instruction="""
    你是一名严谨的事实核查员。
    1. 阅读状态键'draft_text' 中的文本。
    2. 仔细核查所有事实性表述。
    3. 最终输出必须为包含两个键的字典：
    ‑ "status"：字符串，"ACCURATE" 或"INACCURATE"。
    ‑ "reasoning"：字符串，清晰解释你的判断，若有问题需具体说明。
    """,
    output_key="review_output",  # 结构化字典保存到此
)

# SequentialAgent 保证generator 先运行，reviewer 后运行
review_pipeline = SequentialAgent(
    name="WriteAndReview_Pipeline",
    sub_agents=[generator, reviewer],
)
