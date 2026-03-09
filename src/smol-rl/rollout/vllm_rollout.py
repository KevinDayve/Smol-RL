import gc
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from ..config import RolloutConfig
from ..data.experience import Experience
from ..rewards.composite import CompositeReward
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def format_prompt(tokeniser: PreTrainedTokenizer, question: str) -> str:
    """
    Apply model's chat template with system prompt.
    """
    return tokeniser.apply_chat_template(
        [
            {
                "role": "system", "content": SYSTEM_PROMPT
            },
            {
                "role": "user", "content": question
            }
        ],
        tokenize = False,
        add_generation_prompt = True,
    )

class VLLMRolloutManager:
    """
    Handles the full roll-out cycle.
    - Engine creation and teardown.
    - Weight sync from training to vLLM
    - Generation of grouped Completions
    - Reward scoring and construction of experience.
    """
    def __init__(self, model_name_or_path: str, rollout_cfg: RolloutConfig) -> None:
        """
        Args:
            model_name_or_path: HF model ID or local path for engine building.
            rollout_cfg = Typed rolloutd.
        """
        self.config = rollout_cfg
        self.engine: Optional[LLM] = None
        self._sync_dir = Path(".vllm_weights")
        self._create_engine(model_name_or_path)
    def _create_engine(self, model_path: str) -> None:
        """
        Spawn a vLLM engine.
        We enforce eager because we're creating and destroying engines repeatedly. CUDA graph capture aint worth it for short-lived engines.
        """
        self.engine = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_length,
            enforce_eager=True,
        )
    def _destroy_engine(self) -> None:
        """
        Tear down the vLLM engine.
        """
        if self.engine is not None:
            del self.engine
            self.engine = None
            gc.collect()
            torch.cuda.empty_cache()
        
    def sync_weights(self, model: PreTrainedModel, tokeniser: PreTrainedTokenizer) -> None:
        """
        Push current weights into vLLM.
        This goes through the disk which is usually slow;

        To-Do:
            look into model.load_weights() or run vllm on a seperate gpu.
        """
        self._destroy_engine()
        self._sync_dir.mkdir(exist_ok=True)
        model.save_pretrained(self._sync_dir)
        tokeniser.save_pretrained(self._sync_dir)
        self._create_engine(str(self._sync_dir))
    def generate(self, tokeniser: PreTrainedTokenizer, questions: List[str], references: List[str], reward_function: CompositeReward) -> List[Tuple[Experience, List[str]]]:
        """
        Grouped rollouts, with scoring.
        Args:
            tokeniser: The tokeniser for prompt formatting and encoding.
            questions: Batch of question strings.
            referList of (Experience, completions) tuples, one per question.
            Experience has sequences, action_mask, and returns filled in.
            completions is the list of raw strings for logging.ences: Batch of ground-truth answer strings.
            reward_function: Composite reward scorer.
        Returns:
            List of (Experience, completions) tuples, one per question.
            Experience has sequences, action_mask, and returns filled in.
            completions is the list of raw strings for logging.
        """
        input_ids: List[torch.Tensor] = []
        completion_ids: List[torch.Tensor] = []
        prompts = [format_prompt(tokeniser=tokeniser, question=question) for question in questions]
        
        # sampling params
        sampling_params = [
            SamplingParams(
                n=self.config.group_size,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=max(self.config.max_length - len(tokeniser.encode(p)), 1),
            )
            for p in prompts
        ]

        outputs = self.engine.generate(prompts, sampling_params)
        pad_token_id = tokeniser.eos_token_id
        results = []
        for output, reference in zip(outputs, references):
            prompts_ids = list(output.prompt_token_ids)
            prompt_len = len(prompts_ids)

            sequences = []
            completions = []
            rewards = []

            for completion in output.outputs:
                full_ids = prompts_ids + len(completion.token_ids)
                sequences.append(full_ids)
                completions.append(completion.text)
                rewards.append(reward_function.score(completion=completion, reference=reference))
            
            max_seq_length = max(len(s) for s in sequences)
            sequence_ids = torch.full(
                (self.config.group_size, max_seq_length),
                pad_token_id,
                dtype=torch.long
            )
            action_mask = torch.zeros(
                self.config.group_size, max_seq_length-1,
                dtype=torch.bool
            )

            for i, seq in enumerate(sequences):
                sequence_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                action_mask[i, prompt_len - 1 : len(seq) - 1] = True

            returns = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)

            experience = Experience(
                sequences=sequence_ids,
                action_log_probs=None,
                ref_log_probs=None,
                action_mask=action_mask,
                returns=returns,
                advantages=None,
                attention_mask=None,
                kl=None,
            )

        results.append((experience, completions))
        return results