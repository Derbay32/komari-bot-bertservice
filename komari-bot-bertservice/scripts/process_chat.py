"""聊天记录处理脚本

处理 QQ 群聊天记录导出的 JSON 文件：
1. 只保留 type 为 text 的纯文本消息
2. 过滤无效文本（指令、@提及、CQ码等）
3. 合并单个用户 15 秒内发送的连续消息
4. 导出为简化的 JSON 格式
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def should_skip_text(text: str) -> bool:
    """判断文本是否应该被跳过

    Args:
        text: 要检查的文本

    Returns:
        True 表示应该跳过，False 表示保留
    """
    if not text or not text.strip():
        return True

    text = text.strip()

    # 1. 过滤指令文本（如"。jrhg"）
    if re.match(r"^。[a-zA-Z]+$", text):
        return True

    # 2. 过滤包含 @ 的文本
    if "@" in text:
        return True

    # 3. 过滤 CQ 码（如"[图片: xxx.jpg]"）
    if "[图片:" in text or "[语音:" in text or "[视频:" in text:
        return True

    return False


def process_chat_json(
    input_file: Path, output_file: Path, merge_threshold_ms: int = 15000
):
    """处理聊天记录 JSON 文件

    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径
        merge_threshold_ms: 合并阈值（毫秒），默认 15 秒
    """
    print(f"读取文件: {input_file}")

    # 读取原始 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    print(f"原始消息数量: {len(messages)}")

    # 过滤纯文本消息 (type_1)
    text_messages = [
        msg
        for msg in messages
        if msg.get("type") == "type_1"
        and not msg.get("recalled", False)  # 排除撤回的消息
        and not msg.get("system", False)  # 排除系统消息
    ]
    print(f"纯文本消息数量: {len(text_messages)}")

    # 按用户分组
    messages_by_user = defaultdict(list)
    for msg in text_messages:
        sender_uid = msg.get("sender", {}).get("uid", "")
        sender_name = msg.get("sender", {}).get("name", "未知用户")
        timestamp = msg.get("timestamp", 0)
        text = msg.get("content", {}).get("text", "")

        messages_by_user[sender_uid].append(
            {"timestamp": timestamp, "sender_name": sender_name, "text": text}
        )

    # 对每个用户的消息按时间排序
    for uid in messages_by_user:
        messages_by_user[uid].sort(key=lambda x: x["timestamp"])

    # 合并短时间内的连续消息
    merged_messages = []

    for uid, user_messages in messages_by_user.items():
        if not user_messages:
            continue

        # 找到第一条有效消息作为批次起点
        current_batch = []
        current_batch_start = None
        sender_name = None

        for msg in user_messages:
            text = msg["text"]

            # 跳过无效文本
            if should_skip_text(text):
                continue

            # 初始化批次
            if not current_batch:
                current_batch = [msg]
                current_batch_start = msg["timestamp"]
                sender_name = msg["sender_name"]
                continue

            time_diff = msg["timestamp"] - current_batch_start

            # 如果时间差小于阈值，合并到当前批次
            if time_diff <= merge_threshold_ms:
                current_batch.append(msg)
                # 更新批次开始时间为最后一条消息的时间
                current_batch_start = msg["timestamp"]
            else:
                # 时间差超过阈值，保存当前批次，开始新批次
                merged_text = "，".join(m["text"] for m in current_batch)
                merged_messages.append(
                    {"sender_name": sender_name, "text": merged_text}
                )

                # 开始新批次
                current_batch = [msg]
                current_batch_start = msg["timestamp"]
                sender_name = msg["sender_name"]

        # 处理最后一个批次
        if current_batch:
            merged_text = "，".join(m["text"] for m in current_batch)
            merged_messages.append({"sender_name": sender_name, "text": merged_text})

    print(f"合并后消息数量: {len(merged_messages)}")

    # 构建输出数据
    output_data = {"messages": merged_messages}

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"输出文件: {output_file}")
    print("处理完成!")


def main():
    parser = argparse.ArgumentParser(description="处理 QQ 群聊天记录 JSON 文件")
    parser.add_argument("input_file", type=Path, help="输入 JSON 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_file",
        help="输出 JSON 文件路径（默认：输入文件名_processed.json）",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=15,
        dest="threshold_seconds",
        help="合并阈值（秒），默认 15 秒",
    )

    args = parser.parse_args()

    # 确定输出文件名
    if args.output_file is None:
        args.output_file = (
            args.input_file.parent / f"{args.input_file.stem}_processed.json"
        )

    # 转换为毫秒
    merge_threshold_ms = args.threshold_seconds * 1000

    # 处理文件
    try:
        process_chat_json(args.input_file, args.output_file, merge_threshold_ms)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
