import torch
import re
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from maze_env import MazeEnv
from maze_verifier import MazeVerifier
from base.data import Data
from base.verifier import Verifier
from base.env import Env

random.seed(37)

env = MazeEnv()

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

maze_env = MazeEnv()

def create_grpo_dataset(data_list: List[Data]) -> Dataset:
  """
  Converting List[Data] to Dtaset for GRPOTrainer
  """
  prompts = []
  answers = []
  for data in data_list:
    full_prompt = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": data.to_json_str()}
    ]
    prompts.append(full_prompt)
    answers.append(data.to_json_str())

  return Dataset.from_dict({
  "prompt": prompts,
  "answer": answers
  })

def format_reward_func(completions, **kwargs) -> List[float]:
  """
  Reward for XML format
  """
  pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
  responses = [completion[0]["content"] for completion in completions]
  return [1.0 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]

def extract_answer_from_response(response, **kwargs) -> str:
  match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
  return match.group(1).strip() if match else response

def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
  """
  Reward for correct answer
  Compare test answer with gold answer
  """
  responses = [completion[0]["content"] for completion in completions]
  rewards = []
  
  for response_text, data_json in zip(responses, answer):
    data = Data.from_json_str(data_json)
    response_answer = extract_answer_from_response(response_text)
    is_correct = maze_env.verify(data, response_answer)
    rewards.append(1.0 if is_correct else 0.0)
  
  return rewards

train_generator_params = [(1, 300), (2, 400), (3, 500)]

train_dataset = []
for difficulty, num_samples in train_generator_params:
  train_dataset.extend(maze_env.generate(num_samples, difficulty))
  
train_dataset = create_grpo_dataset(train_dataset)

test_dataset = {
  "Easy": maze_env.generate(50, difficulty=1),
  "Medium": maze_env.generate(75, difficulty=2),
  "Hard": maze_env.generate(100, difficulty=3)
}

max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "Qwen/Qwen2.5-3B-Instruct",
  max_seq_length = max_seq_length, 
  load_in_4bit = True,
  max_lora_rank = lora_rank,
  gpu_memory_utilization = 0.9
)

model = FastLanguageModel.get_peft_model(
  model,
  r = lora_rank,
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj",],
  lora_alpha = lora_rank,
  use_gradient_checkpointing = "unsloth", 
  random_state = 3407
)

training_args = GRPOConfig(
  output_dir = "outputs/qwen-maze-grpo",
  learning_rate = 5e-6,
  adam_beta1 = 0.9,
  adam_beta2 = 0.99,
  weight_decay = 0.1,
  warmup_ratio = 0.1,
  lr_scheduler_type = "cosine",
  logging_steps = 10,
  bf16 = True,
  per_device_train_batch_size = 24,
  gradient_accumulation_steps = 1,
  num_generations = 2, 
  max_prompt_length = 512,
  max_completion_length = 150,
  num_train_epochs = 1, 
  save_steps = 100,
  max_grad_norm = 0.1,
  report_to = "none",
)

trainer = GRPOTrainer(
  model = model,
  processing_class = tokenizer,
  reward_funcs = [format_reward_func, correctness_reward_func],
  args = training_args,
  train_dataset = train_dataset,
)

trainer.train()