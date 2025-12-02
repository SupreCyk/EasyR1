# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import base64
import asyncio
import time
import hashlib
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from PIL import Image

# 导入prompts
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from prompts.select_rubric import (
    SYSTEM_PROMPT as SELECT_SYSTEM_PROMPT,
    USER_PROMPT as SELECT_USER_PROMPT,
)
from prompts.dynamic import (
    SYSTEM_PROMPT as DYNAMIC_SYSTEM_PROMPT,
    USER_PROMPT as DYNAMIC_USER_PROMPT,
)
from prompts.factors import FACTORS

# Metadata
REWARD_NAME = "dynamic_rubric_api"
REWARD_TYPE = "batch"

# API配置
API_ENDPOINT = "https://api.a1r.cc/v1/chat/completions"
API_KEY = "sk-cDNrDh4dQfjiVnyby9B4K3NefSLvFhRlbbVMg3pqhKS1707p"  # 替换为你的实际AK
MODEL_NAME = "gpt-4.1"  # 可以配置的模型名称

def _parse_factors(factors_text: str) -> Dict[str, Dict[str, str]]:
    """将factors描述解析为字典"""
    entries: Dict[str, Dict[str, str]] = {}
    current_name: Optional[str] = None
    description_lines: List[str] = []
    scoring_line: Optional[str] = None

    def _flush() -> None:
        nonlocal current_name, description_lines, scoring_line
        if current_name:
            entries[current_name] = {
                "description": " ".join(description_lines).strip(),
                "scoring": (scoring_line or "").strip(),
            }
        current_name = None
        description_lines = []
        scoring_line = None

    for raw_line in factors_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header_match = re.match(r"^\d+\.\s*([A-Za-z0-9_]+)", line)
        if header_match:
            _flush()
            current_name = header_match.group(1)
            continue
        normalized = line.lstrip("- ").strip()
        if normalized.lower().startswith("score"):
            scoring_line = normalized
        else:
            description_lines.append(normalized)

    _flush()
    return entries


RUBRIC_INFOS: Dict[str, Dict[str, str]] = _parse_factors(FACTORS)

FALLBACK_SELECTED_RUBRICS: List[Dict[str, float]] = [
    {"name": "correctness_numeric", "weight": 0.7},
    {"name": "math_validity", "weight": 0.2},
    {"name": "visual_interpretation", "weight": 0.1},
]

_RUBRIC_SELECTION_CACHE: Dict[str, Dict[str, Any]] = {}

# 默认分数（用于失败情况）
DEFAULT_SCORES = {
    "correctness_numeric": 0.0,
    "visual_interpretation": 0.0,
    "math_validity": 0.0,
    "instruction_following": 0.0,
    "expression_format": 0.0,
}


def _extract_raw_api_response(api_info: Any) -> Optional[str]:
    """从API信息中提取原始字符串表示"""
    if api_info is None:
        return None
    if isinstance(api_info, dict):
        raw = api_info.get("api_raw_response")
        if raw is not None:
            return raw
        try:
            return json.dumps(api_info, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(api_info)
    return str(api_info)


def image_to_data_uri(image) -> str:
    """将PIL Image或图像路径转换为data URI格式"""
    if isinstance(image, str):
        # 如果是文件路径
        with open(image, "rb") as f:
            image_data = f.read()
            base64_str = base64.b64encode(image_data).decode("utf-8")
            # 尝试根据文件扩展名判断MIME类型
            if image.lower().endswith('.png'):
                mime_type = "image/png"
            elif image.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"  # 默认
            return f"data:{mime_type};base64,{base64_str}"
    elif isinstance(image, Image.Image):
        # 如果是PIL Image对象
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def build_selection_key(reward_input: Dict[str, Any]) -> str:
    """优先使用uid生成rubric选择缓存键，缺失时退回问题hash"""
    metadata = reward_input.get("metadata", {}) if isinstance(reward_input, dict) else {}
    uid = metadata.get("uid") or reward_input.get("uid")
    if uid:
        return str(uid)
    problem = reward_input.get("problem", "")
    ground_truth = reward_input.get("ground_truth", "")
    raw = f"{problem}##{ground_truth}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sanitize_selected_rubrics(
    selected_rubrics: List[Dict[str, Any]],
    rubric_count: int,
) -> List[Dict[str, float]]:
    """过滤非法rubric并对权重归一化"""
    valid: List[Dict[str, float]] = []
    for entry in selected_rubrics:
        name = entry.get("name") or entry.get("id")
        weight = entry.get("weight", 0.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 0.0
        if name in RUBRIC_INFOS and weight > 0:
            valid.append({"name": name, "weight": weight})
    if not valid:
        return FALLBACK_SELECTED_RUBRICS[:rubric_count]
    valid.sort(key=lambda x: x["weight"], reverse=True)
    valid = valid[:rubric_count]
    total = sum(item["weight"] for item in valid)
    if total <= 0:
        return FALLBACK_SELECTED_RUBRICS[:rubric_count]
    return [{"name": item["name"], "weight": item["weight"] / total} for item in valid]


def build_rubric_block(selected_rubrics: List[Dict[str, float]]) -> str:
    """将rubric定义渲染成模板需要的文本"""
    lines: List[str] = []
    for idx, entry in enumerate(selected_rubrics, start=1):
        name = entry["name"]
        info = RUBRIC_INFOS.get(name, {})
        description = info.get("description", "")
        scoring = info.get("scoring", "")
        lines.append(f"{idx}. {name}")
        if description:
            lines.append(f"   - {description}")
        if scoring:
            lines.append(f"   - {scoring}")
    return "\n".join(lines)


async def call_rubric_selection_api_async(
    session: aiohttp.ClientSession,
    image_data_uri: str,
    problem_text: str,
    ground_truth: str,
    api_endpoint: str,
    api_key: str,
    model_name: str,
    timeout: int,
    max_tokens: int,
    max_retries: int,
    idx: str,
    rubric_count: int,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """调用rubric选择API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    user_text = SELECT_USER_PROMPT.format(
        IMAGE_CONTENT="See attached image.",
        PROBLEM_TEXT=problem_text,
        GROUND_TRUTH=ground_truth,
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SELECT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri,},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with session.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_obj,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                if "choices" not in result or not result["choices"]:
                    raise ValueError("selection API returned empty choices")
                content = result["choices"][0]["message"]["content"]
                api_info = {
                    "api_raw_response": content,
                    "api_model": model_name,
                    "api_usage": result.get("usage", {}),
                    "api_finish_reason": result["choices"][0].get("finish_reason", ""),
                    "api_stage": "selection",
                }
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"selection response is not JSON: {content[:200]}") from exc
                rubrics = parsed.get("selected_rubrics") or parsed.get("rubrics") or []
                sanitized = sanitize_selected_rubrics(rubrics, rubric_count)
                return sanitized, api_info
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            print(f"rubric selection {idx}: network error attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            error_info = {
                "api_error": "selection_timeout" if isinstance(exc, asyncio.TimeoutError) else "selection_network_error",
                "api_stage": "selection",
            }
            return FALLBACK_SELECTED_RUBRICS[:rubric_count], error_info
        except Exception as exc:
            print(f"rubric selection {idx}: error attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            error_info = {"api_error": "selection_unknown_error", "api_stage": "selection"}
            return FALLBACK_SELECTED_RUBRICS[:rubric_count], error_info
    return FALLBACK_SELECTED_RUBRICS[:rubric_count], {"api_stage": "selection"}


async def call_dynamic_evaluation_api_async(
    session: aiohttp.ClientSession,
    image_data_uri: str,
    problem_text: str,
    ground_truth: str,
    model_output: str,
    selected_rubrics: List[Dict[str, float]],
    api_endpoint: str,
    api_key: str,
    model_name: str,
    timeout: int,
    max_tokens: int,
    max_retries: int,
    idx: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """只针对选定rubric进行评分"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    rubric_names = [item["name"] for item in selected_rubrics]
    rubric_block = build_rubric_block(selected_rubrics)
    # system模板包含JSON示例的大括号，不能直接用str.format；仅替换目标占位符
    system_prompt = DYNAMIC_SYSTEM_PROMPT.replace("{RUBRIC_INFOS}", rubric_block)
    user_text = DYNAMIC_USER_PROMPT.format(
        PROBLEM_TEXT=problem_text,
        GROUND_TRUTH=ground_truth,
        MODEL_OUTPUT=model_output,
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with session.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_obj,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                if "choices" not in result or not result["choices"]:
                    raise ValueError("evaluation API returned empty choices")
                content = result["choices"][0]["message"]["content"]
                api_info = {
                    "api_raw_response": content,
                    "api_model": model_name,
                    "api_usage": result.get("usage", {}),
                    "api_finish_reason": result["choices"][0].get("finish_reason", ""),
                    "api_stage": "evaluation",
                }
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    import re

                    match = re.search(r"\{[\s\S]*\}", content)
                    if not match:
                        raise ValueError(f"evaluation response not JSON: {content[:200]}") from exc
                    parsed = json.loads(match.group(0))
                scores = parsed.get("scores", parsed)
                filtered_scores = {name: float(scores.get(name, 0.0)) for name in rubric_names}
                return filtered_scores, api_info
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            print(f"evaluation {idx}: network error attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            error_info = {
                "api_error": "evaluation_timeout" if isinstance(exc, asyncio.TimeoutError) else "evaluation_network_error",
                "api_stage": "evaluation",
            }
            return {name: 0.0 for name in rubric_names}, error_info
        except Exception as exc:
            print(f"evaluation {idx}: error attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            error_info = {"api_error": "evaluation_unknown_error", "api_stage": "evaluation"}
            return {name: 0.0 for name in rubric_names}, error_info
    return {name: 0.0 for name in rubric_names}, {"api_stage": "evaluation"}


async def select_rubrics_for_batch(
    prepared_inputs: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    api_endpoint: str,
    api_key: str,
    model_name: str,
    timeout: int,
    max_retries: int,
    max_concurrent: int,
    rubric_count: int,
) -> Dict[str, Dict[str, Any]]:
    """针对每个uid/index选择rubric"""
    results: Dict[str, Dict[str, Any]] = {}
    unique_entries: Dict[str, Dict[str, Any]] = {}
    for item in prepared_inputs:
        key = item["selection_key"]
        if key not in unique_entries:
            unique_entries[key] = item
    pending = []
    for key, entry in unique_entries.items():
        cached = _RUBRIC_SELECTION_CACHE.get(key)
        if cached:
            results[key] = cached
            continue
        if entry["image_data_uri"] is None:
            info = {
                "rubrics": FALLBACK_SELECTED_RUBRICS[:rubric_count],
                "api_info": {"api_error": "no_image_for_selection", "api_stage": "selection"},
            }
            results[key] = info
            _RUBRIC_SELECTION_CACHE[key] = info
            continue
        pending.append((key, entry))
    semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def process_pending(key: str, entry: Dict[str, Any]) -> None:
        async with semaphore:
            rubrics, api_info = await call_rubric_selection_api_async(
                session=session,
                image_data_uri=entry["image_data_uri"],
                problem_text=entry["problem"],
                ground_truth=entry["ground_truth"],
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout,
                max_tokens=400,
                max_retries=max_retries,
                idx=key,
                rubric_count=rubric_count,
            )
        info = {"rubrics": rubrics, "api_info": api_info}
        results[key] = info
        _RUBRIC_SELECTION_CACHE[key] = info

    if pending:
        await asyncio.gather(*(process_pending(key, entry) for key, entry in pending))
    return results


async def process_batch_async(
    reward_inputs: List[Dict[str, Any]],
    api_endpoint: str,
    api_key: str,
    model_name: str,
    max_concurrent: int = 50,
    timeout: int = 60,
    max_retries: int = 3,
    selector_max_concurrent: int = 10,
    selector_timeout: int = 40,
    selector_max_retries: int = 3,
    rubric_count: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    异步批量处理动态rubric reward计算
    """
    prepared_inputs: List[Dict[str, Any]] = []
    for idx, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        problem = reward_input.get("problem", "")
        ground_truth = reward_input.get("ground_truth", "")
        multi_modal_data = reward_input.get("multi_modal_data")
        image_data_uri: Optional[str] = None
        if isinstance(multi_modal_data, dict):
            images = multi_modal_data.get("images", [])
            if images:
                try:
                    image_data_uri = image_to_data_uri(images[0])
                except Exception as exc:
                    print(f"样本 {idx}: 图像转换失败 {exc}")
        prepared_inputs.append(
            {
                "idx": idx,
                "response": response,
                "problem": problem,
                "ground_truth": ground_truth,
                "image_data_uri": image_data_uri,
                "selection_key": build_selection_key(reward_input),
            }
        )

    connector = aiohttp.TCPConnector(limit=max(max_concurrent, selector_max_concurrent), limit_per_host=0)
    timeout_obj = aiohttp.ClientTimeout(total=max(timeout, selector_timeout) * 2)

    unique_selection_keys = {item["selection_key"] for item in prepared_inputs}
    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
        print(
            f"开始选择rubric，唯一样本数 {len(unique_selection_keys)}，最大并发 {selector_max_concurrent}"
        )
        selection_start = time.time()
        selection_results = await select_rubrics_for_batch(
            prepared_inputs=prepared_inputs,
            session=session,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_name=model_name,
            timeout=selector_timeout,
            max_retries=selector_max_retries,
            max_concurrent=selector_max_concurrent,
            rubric_count=rubric_count,
        )
        selection_elapsed = time.time() - selection_start
        avg_selection = selection_elapsed / max(1, len(unique_selection_keys))
        print(
            f"rubric选择完成，耗时 {selection_elapsed:.2f}s，平均 {avg_selection:.2f}s/样本"
        )

        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        async def process_single(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            async with semaphore:
                selected_info = selection_results.get(
                    item["selection_key"],
                    {"rubrics": FALLBACK_SELECTED_RUBRICS[:rubric_count], "api_info": None},
                )
                selected_rubrics = selected_info["rubrics"]
                selection_info_raw = _extract_raw_api_response(selected_info.get("api_info"))

                if item["image_data_uri"] is None:
                    judge_info_raw = "no_image"
                    return (
                        {"scores": DEFAULT_SCORES.copy(), "selected_rubrics": selected_rubrics},
                        {
                            "api_judge_info": judge_info_raw,
                            "api_selection_info": selection_info_raw,
                            "selected_rubrics": selected_rubrics,
                            "rubric_scores": DEFAULT_SCORES.copy(),
                        },
                    )

                rubric_scores, api_response_info = await call_dynamic_evaluation_api_async(
                    session=session,
                    image_data_uri=item["image_data_uri"],
                    problem_text=item["problem"],
                    ground_truth=item["ground_truth"],
                    model_output=item["response"],
                    selected_rubrics=selected_rubrics,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    model_name=model_name,
                    timeout=timeout,
                    max_tokens=600,
                    max_retries=max_retries,
                    idx=item["idx"],
                )
                return (
                    {"scores": rubric_scores, "selected_rubrics": selected_rubrics},
                    {
                        "api_judge_info": _extract_raw_api_response(api_response_info),
                        "api_selection_info": selection_info_raw,
                        "selected_rubrics": selected_rubrics,
                        "rubric_scores": rubric_scores,
                    },
                )

        print(f"开始动态rubric评估，样本数 {len(prepared_inputs)}，评分最大并发 {max_concurrent}")
        start_time = time.time()
        results = await asyncio.gather(*(process_single(item) for item in prepared_inputs))
        elapsed = time.time() - start_time
        print(f"动态rubric评估完成，耗时 {elapsed:.2f}s，平均 {elapsed / max(1, len(prepared_inputs)):.2f}s/样本")

    rubric_scores_list = [entry[0] for entry in results]
    api_response_list = [entry[1] for entry in results]
    return rubric_scores_list, api_response_list


def compute_weighted_score(rubric_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算加权平均分数"""
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    total_score = 0.0
    for rubric, weight in weights.items():
        total_score += rubric_scores.get(rubric, 0.0) * weight
    return total_score / total_weight


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    api_endpoint: str = API_ENDPOINT,
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
    max_concurrent: int = 50,
    timeout: int = 60,
    max_retries: int = 3,
    selector_max_concurrent: int = 10,
    selector_timeout: int = 40,
    selector_max_retries: int = 3,
    rubric_count: int = 3,
    fallback_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    批量计算reward分数（同步接口，内部使用异步实现）
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            rubric_scores_list, api_response_list = loop.run_until_complete(
                process_batch_async(
                    reward_inputs=reward_inputs,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    model_name=model_name,
                    max_concurrent=max_concurrent,
                    timeout=timeout,
                    max_retries=max_retries,
                    selector_max_concurrent=selector_max_concurrent,
                    selector_timeout=selector_timeout,
                    selector_max_retries=selector_max_retries,
                    rubric_count=rubric_count,
                )
            )
        else:
            rubric_scores_list, api_response_list = loop.run_until_complete(
                process_batch_async(
                    reward_inputs=reward_inputs,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    model_name=model_name,
                    max_concurrent=max_concurrent,
                    timeout=timeout,
                    max_retries=max_retries,
                    selector_max_concurrent=selector_max_concurrent,
                    selector_timeout=selector_timeout,
                    selector_max_retries=selector_max_retries,
                    rubric_count=rubric_count,
                )
            )
    except RuntimeError:
        rubric_scores_list, api_response_list = asyncio.run(
            process_batch_async(
                reward_inputs=reward_inputs,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                max_concurrent=max_concurrent,
                timeout=timeout,
                max_retries=max_retries,
                selector_max_concurrent=selector_max_concurrent,
                selector_timeout=selector_timeout,
                selector_max_retries=selector_max_retries,
                rubric_count=rubric_count,
            )
        )

    scores: List[Dict[str, Any]] = []
    for sample in rubric_scores_list:
        rubric_scores = sample.get("scores", {})
        selected_rubrics = sample.get("selected_rubrics", FALLBACK_SELECTED_RUBRICS[:rubric_count])
        weights = {item["name"]: item["weight"] for item in selected_rubrics}
        if not weights and fallback_weights:
            weights = fallback_weights
        elif not weights:
            weights = {item["name"]: item["weight"] for item in FALLBACK_SELECTED_RUBRICS[:rubric_count]}
        overall_score = compute_weighted_score(rubric_scores, weights)
        scores.append({
            "overall": overall_score,
            **rubric_scores,
        })
    return scores, api_response_list