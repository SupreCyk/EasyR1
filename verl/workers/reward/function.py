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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer
import numpy as np

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]], Optional[DataProto]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        # 对于sequential模式，也返回data以保持接口一致（虽然没有额外信息）
        return reward_tensor, reward_metrics, None


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]], Optional[DataProto]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            # 构建reward input，添加更多字段
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            
            # 添加原始问题文本
            # if "problem" in data.non_tensor_batch:
            #     reward_input["problem"] = data.non_tensor_batch["problem"][i]
            # 添加原始问题文本（移除多模态标记用于API调用）
            if "problem" in data.non_tensor_batch:
                problem_text = str(data.non_tensor_batch["problem"][i])
                # 清理 <image> 和 <video> 标记，只保留纯文本
                problem_text = problem_text.replace("<image>", "").replace("<video>", "").strip()
                reward_input["problem"] = problem_text
            
            # 添加图像数据
            if "multi_modal_data" in data.non_tensor_batch and data.non_tensor_batch["multi_modal_data"][i] is not None:
                reward_input["multi_modal_data"] = data.non_tensor_batch["multi_modal_data"][i]
            
            reward_inputs.append(reward_input)

        # 调用reward函数，可能返回分数和额外信息
        result = self.reward_fn(reward_inputs)
        
        # 检查返回值类型
        if isinstance(result, tuple) and len(result) == 2:
            scores, api_response_list = result
            # 将API响应信息添加到non_tensor_batch中，使用 numpy.array 而不是 torch.tensor
            try:
                judge_infos = []
                selection_infos = []
                selected_rubrics_list = []
                rubric_scores_list = []
                fallback_infos = []
                for entry in api_response_list:
                    if isinstance(entry, dict):
                        judge_infos.append(entry.get("api_judge_info"))
                        selection_infos.append(entry.get("api_selection_info"))
                        selected_rubrics_list.append(entry.get("selected_rubrics"))
                        rubric_scores_list.append(entry.get("rubric_scores"))
                        fallback_infos.append(entry)
                    else:
                        judge_infos.append(None)
                        selection_infos.append(None)
                        selected_rubrics_list.append(None)
                        rubric_scores_list.append(None)
                        fallback_infos.append(entry)
                data.non_tensor_batch["api_judge_info"] = np.array(judge_infos, dtype=object)
                data.non_tensor_batch["api_selection_info"] = np.array(selection_infos, dtype=object)
                data.non_tensor_batch["selected_rubrics"] = np.array(selected_rubrics_list, dtype=object)
                data.non_tensor_batch["rubric_scores"] = np.array(rubric_scores_list, dtype=object)
                data.non_tensor_batch["api_response_info"] = np.array(fallback_infos, dtype=object)
            except Exception:
                # 兼容旧格式，直接存储
                data.non_tensor_batch["api_response_info"] = np.array(api_response_list, dtype=object)
        else:
            scores = result
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        # 返回修改后的data，以便传递api_response_info等非张量数据
        return reward_tensor, reward_metrics, data


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]], Optional[DataProto]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
