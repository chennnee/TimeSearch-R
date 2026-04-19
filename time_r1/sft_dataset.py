import os
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# 读取方舟 API Key
API_KEY = "ark-56387331-0087-4cf2-b42b-f5526c887d11-172a4"
if not API_KEY:
    raise ValueError("请先设置环境变量 ARK_API_KEY")

# 初始化 Ark 客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=API_KEY,
)

SYSTEM_PROMPT = """你是一个长视频推理数据标注助手。
你的任务是根据视频和问题，生成用于监督微调（SFT）的训练样本。

你必须严格输出如下格式，且只能输出这一种格式，不要加任何解释：

<think>...</think>
<tool_call>{"start_time": 数字, "end_time": 数字, "query": "字符串", "num_frames": 8}</tool_call>
<think>...</think>
<tool_call>{"start_time": 数字, "end_time": 数字, "query": "字符串", "num_frames": 8}</tool_call>
...
<think>...</think>
<answer>最终答案</answer>

要求：
1. 最多 4 次 <tool_call>
2. 每个 tool_call 都必须是合法 JSON
3. start_time <= end_time
4. query 要简洁，像检索词，不要写成长句
5. 最后必须以 <answer> 结束
6. 如果是选择题，answer 只输出选项字母，如 A / B / C / D
7. 不要输出 markdown 代码块，不要输出多余说明
"""

USER_TEMPLATE = """问题：
{question}

补充信息：
- 视频总时长：{duration_hint}
- 题目类型：{qtype}
- 如果有选项，请结合选项作答：{options}

请生成一条高质量的“视频搜索 + 推理 + 最终答案”的 SFT 标注。
"""

def build_user_message(
    question: str,
    options: Optional[str],
    video_url: str,
    duration_hint: str = "未知",
    qtype: str = "multiple_choice",
    fps: int = 2,
):
    content = [
        {
            "type": "video_url",
            "video_url": {
                "url": video_url
            },
            "fps": str(fps)
        },
        {
            "type": "text",
            "text": USER_TEMPLATE.format(
                question=question,
                duration_hint=duration_hint,
                qtype=qtype,
                options=options if options else "无",
            )
        }
    ]

    return {
        "role": "user",
        "content": content
    }

def call_ark_sft(
    question: str,
    options: Optional[str],
    video_url: str,
    duration_hint: str = "未知",
    qtype: str = "multiple_choice",
    model: str = "ep-20260419115315-z9zlw",
    fps: int = 2,
):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        build_user_message(
            question=question,
            options=options,
            video_url=video_url,
            duration_hint=duration_hint,
            qtype=qtype,
            fps=fps,
        )
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0.2,
    )

    return resp.choices[0].message.content

def is_valid_format(text: str) -> bool:
    if not text:
        return False
    if "<think>" not in text or "</think>" not in text:
        return False
    if "<answer>" not in text or "</answer>" not in text:
        return False
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return False
    return True

def generate_one_sample(sample: dict, max_retry: int = 3):
    for i in range(max_retry):
        try:
            output = call_ark_sft(
                question=sample["question"],
                options=sample.get("options"),
                video_url=sample["video_url"],
                duration_hint=sample.get("duration_hint", "未知"),
                qtype=sample.get("qtype", "multiple_choice"),
                fps=sample.get("fps", 2),
            )
            if is_valid_format(output):
                return {
                    "id": sample["id"],
                    "video_url": sample["video_url"],
                    "question": sample["question"],
                    "options": sample.get("options"),
                    "answer": sample.get("answer"),
                    "sft_target": output,
                }
            else:
                print(f"[retry {i+1}] {sample['id']} format invalid")
        except Exception as e:
            print(f"[retry {i+1}] {sample['id']} failed: {e}")
            time.sleep(2)

    return {
        "id": sample["id"],
        "video_url": sample["video_url"],
        "question": sample["question"],
        "options": sample.get("options"),
        "answer": sample.get("answer"),
        "sft_target": None,
    }

if __name__ == "__main__":
    demo_samples = [
        {
            "id": "demo-0001",
            "video_url": "https://chen-2026yolo-data.aoss.cn-sh-01g.sensecoreapi-oss.cn/redpandacompress_video1.mp4",
            "question": "What does the woman do near the end of the video?",
            "options": "A. She sits back on the sofa\nB. She stands up from the sofa\nC. She opens the curtain\nD. She picks up a phone",
            "answer": "B",
            "duration_hint": "40.7 秒",
            "qtype": "multiple_choice",
            "fps": 2
        }
    ]

    results = []
    for s in demo_samples:
        item = generate_one_sample(s)
        results.append(item)
        print("=" * 80)
        print(item["id"])
        print(item["sft_target"])

    out_path = Path("ark_seed_sft_annotations.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"saved to: {out_path}")