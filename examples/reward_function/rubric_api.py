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
from io import BytesIO
from typing import Any, Dict, List, Optional
import aiohttp
from PIL import Image

# 导入prompts
import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from prompts.static import SYSTEM_PROMPT, USER_PROMPT

# Metadata
REWARD_NAME = "rubric_api"
REWARD_TYPE = "batch"

# Rubric权重配置
DEFAULT_RUBRIC_WEIGHTS = {
    "correctness_numeric": 0.7,
    "visual_interpretation": 0.1,
    "math_validity": 0.1,
    "instruction_following": 0.05,
    "expression_format": 0.05,
}

# API配置
API_ENDPOINT = "https://api.a1r.cc/v1/chat/completions"
API_KEY = "sk-cDNrDh4dQfjiVnyby9B4K3NefSLvFhRlbbVMg3pqhKS1707p"  # 替换为你的实际AK
MODEL_NAME = "gpt-4.1-mini"  # 可以配置的模型名称

# 默认分数（用于失败情况）
DEFAULT_SCORES = {
    "correctness_numeric": 0.0,
    "visual_interpretation": 0.0,
    "math_validity": 0.0,
    "instruction_following": 0.0,
    "expression_format": 0.0,
}


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


async def call_evaluation_api_async(
    session: aiohttp.ClientSession,
    image_data_uri: str,
    problem_text: str,
    ground_truth: str,  # 新增参数
    model_output: str,
    api_endpoint: str,
    api_key: str,
    model_name: str,
    timeout: int = 60,
    max_tokens: int = 500,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    idx: int = 0,
) -> tuple[Dict[str, float], Dict[str, Any]]:  # 修改返回类型
    """
    异步调用OpenAI格式的API进行评估，带重试机制
    
    Args:
        session: aiohttp客户端会话
        image_data_uri: 图像的data URI
        problem_text: 问题文本
        ground_truth: 真实答案  # 新增说明
        model_output: 模型输出
        api_endpoint: API端点
        api_key: API密钥
        model_name: 模型名称
        timeout: 超时时间（秒）
        max_tokens: 最大token数
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒），使用指数退避
        idx: 样本索引（用于日志）
    
    Returns:
        包含各个rubric分数的字典 和 API完整响应信息的元组
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # 构建user prompt
    user_text = USER_PROMPT.format(
        PROBLEM_TEXT=problem_text,
        GROUND_TRUTH=ground_truth,  # 新增
        MODEL_OUTPUT=model_output
    )
    
    # 构建OpenAI格式的消息
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri,
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            }
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    
    # 重试循环
    for attempt in range(max_retries):
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with session.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_obj
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                # 解析API返回的内容
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # 构建API响应信息
                    api_response_info = {
                        "api_raw_response": content,
                        "api_model": model_name,
                        "api_usage": result.get("usage", {}),
                        "api_finish_reason": result["choices"][0].get("finish_reason", ""),
                    }
                    
                    # 尝试解析JSON格式的分数
                    try:
                        scores_data = json.loads(content)
                        if "scores" in scores_data:
                            return scores_data["scores"], api_response_info
                        else:
                            return scores_data, api_response_info
                    except json.JSONDecodeError:
                        # 如果不是纯JSON，尝试提取JSON部分
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            scores_data = json.loads(json_match.group())
                            if "scores" in scores_data:
                                return scores_data["scores"], api_response_info
                            else:
                                return scores_data, api_response_info
                        else:
                            print(f"样本 {idx}: 无法解析API返回的内容: {content[:200]}...")
                            raise ValueError("API返回内容不是有效的JSON格式")
                else:
                    print(f"样本 {idx}: API返回格式异常")
                    raise ValueError("API返回格式不符合预期")
        
        except asyncio.TimeoutError:
            print(f"样本 {idx}: 请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # 指数退避
                await asyncio.sleep(delay)
            else:
                print(f"样本 {idx}: 达到最大重试次数，返回默认分数")
                error_info = {
                    "api_raw_response": "timeout",
                    "api_model": model_name,
                    "api_error": "timeout",
                }
                return DEFAULT_SCORES.copy(), error_info
        
        except aiohttp.ClientError as e:
            print(f"样本 {idx}: 网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                print(f"样本 {idx}: 达到最大重试次数，返回默认分数")
                error_info = {
                    "api_raw_response": "network_error",
                    "api_model": model_name,
                    "api_error": "network_error",
                }
                return DEFAULT_SCORES.copy(), error_info
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"样本 {idx}: 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                print(f"样本 {idx}: 达到最大重试次数，返回默认分数")
                error_info = {
                    "api_raw_response": "parse_error",
                    "api_model": model_name,
                    "api_error": "parse_error",
                }
                return DEFAULT_SCORES.copy(), error_info
        
        except Exception as e:
            print(f"样本 {idx}: 未知错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                print(f"样本 {idx}: 达到最大重试次数，返回默认分数")
                error_info = {
                    "api_raw_response": "unknown_error",
                    "api_model": model_name,
                    "api_error": "unknown_error",
                }
                return DEFAULT_SCORES.copy(), error_info
    
    return DEFAULT_SCORES.copy(), {} # This line should ideally not be reached if max_retries > 0


async def process_batch_async(
    reward_inputs: List[Dict[str, Any]],
    api_endpoint: str,
    api_key: str,
    model_name: str,
    max_concurrent: int = 50,
    timeout: int = 60,
    max_retries: int = 3,
) -> tuple[List[Dict[str, float]], List[Dict[str, Any]]]:  # 修改返回类型
    """
    异步批量处理reward计算
    
    Args:
        reward_inputs: 输入数据列表
        api_endpoint: API端点
        api_key: API密钥
        model_name: 模型名称
        max_concurrent: 最大并发数
        timeout: 单个请求超时时间
        max_retries: 最大重试次数
    
    Returns:
        分数列表 和 API响应信息列表的元组
    """
    # 创建信号量来限制并发
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(idx: int, reward_input: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
        """处理单个样本"""
        async with semaphore:
            response = reward_input["response"]
            problem = reward_input.get("problem", "")
            ground_truth = reward_input.get("ground_truth", "")  # 提取真实答案
            multi_modal_data = reward_input.get("multi_modal_data", None)
            
            # 提取图像
            image_data_uri = None
            if multi_modal_data and "images" in multi_modal_data:
                images = multi_modal_data["images"]
                if len(images) > 0:
                    try:
                        image_data_uri = image_to_data_uri(images[0])
                    except Exception as e:
                        print(f"样本 {idx}: 图像转换失败: {e}")
            
            # 如果没有图像，返回默认分数
            if image_data_uri is None:
                print(f"样本 {idx}: 警告：没有找到图像数据，使用默认分数")
                error_info = {"api_error": "no_image"}
                return DEFAULT_SCORES.copy(), error_info
            
            # 调用API
            print(f"样本 {idx}: 调用API")
            print(f"样本 {idx}: 问题文本: {problem}")
            print(f"样本 {idx}: 真实答案: {ground_truth}")
            print(f"样本 {idx}: 模型输出: {response}")
            rubric_scores, api_response_info = await call_evaluation_api_async(
                session=session,
                image_data_uri=image_data_uri,
                problem_text=problem,
                ground_truth=ground_truth,  # 传入真实答案
                model_output=response,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout,
                max_retries=max_retries,
                idx=idx,
            )
            
            # call_evaluation_api_async 已经返回了处理好的字典，直接返回即可
            return rubric_scores, api_response_info

    # 创建aiohttp会话
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout_obj = aiohttp.ClientTimeout(total=timeout * 2)  # 总超时时间
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
        # 创建所有任务
        tasks = [
            process_single(idx, reward_input)
            for idx, reward_input in enumerate(reward_inputs)
        ]
        
        # 执行所有任务并显示进度
        print(f"开始异步处理 {len(tasks)} 个样本，最大并发数: {max_concurrent}")
        start_time = time.time()
        
        # 使用gather收集所有结果
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        elapsed_time = time.time() - start_time
        print(f"批量处理完成，耗时: {elapsed_time:.2f}秒，平均每个样本: {elapsed_time/len(tasks):.2f}秒")
        
        # 分离分数和API响应信息
        rubric_scores_list = [r[0] for r in results]
        api_response_list = [r[1] for r in results]
        
        return rubric_scores_list, api_response_list


def compute_weighted_score(rubric_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算加权平均分数"""
    total_score = 0.0
    total_weight = sum(weights.values())
    
    for rubric, score in rubric_scores.items():
        weight = weights.get(rubric, 0.0)
        total_score += score * weight
    
    return total_score / total_weight if total_weight > 0 else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    rubric_weights: Dict[str, float] = None,
    api_endpoint: str = API_ENDPOINT,
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
    max_concurrent: int = 50,
    timeout: int = 60,
    max_retries: int = 3,
) -> tuple[List[Dict[str, float]], List[Dict[str, Any]]]:  # 修改返回类型
    """
    批量计算reward分数（同步接口，内部使用异步实现）
    
    Args:
        reward_inputs: 包含以下字段的字典列表
            - response: 模型生成的回复
            - problem: 原始问题文本
            - multi_modal_data: 包含图像的字典，格式为 {"images": [image1, image2, ...]}
            - ground_truth: 标准答案（可选，用于日志）
        rubric_weights: rubric权重字典
        api_endpoint: API端点
        api_key: API密钥
        model_name: 使用的模型名称
        max_concurrent: 最大并发请求数（建议50-100）
        timeout: 单个请求超时时间（秒）
        max_retries: 最大重试次数
    
    Returns:
        包含各个维度分数和overall分数的字典列表 和 API响应信息列表的元组
    """
    if rubric_weights is None:
        rubric_weights = DEFAULT_RUBRIC_WEIGHTS
    
    # 运行异步批处理
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在运行的事件循环中，创建新的事件循环
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
                )
            )
    except RuntimeError:
        # 如果没有事件循环，创建新的
        rubric_scores_list, api_response_list = asyncio.run(
            process_batch_async(
                reward_inputs=reward_inputs,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model_name=model_name,
                max_concurrent=max_concurrent,
                timeout=timeout,
                max_retries=max_retries,
            )
        )
    
    # 计算加权分数并构建返回结果
    scores = []
    for idx, rubric_scores in enumerate(rubric_scores_list):
        overall_score = compute_weighted_score(rubric_scores, rubric_weights)
        
        result = {
            "overall": overall_score,
            **rubric_scores,
        }
        scores.append(result)
    
    return scores, api_response_list  # 返回两个列表