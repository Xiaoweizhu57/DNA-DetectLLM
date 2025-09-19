from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy, min_perplexity, entropy_pro, calculate_window_ppl, calculate_window_entropy, joint_score_ratio, w_entropy, auc_perplexity, relative_entropy
from .metrics import sum_perplexity

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
detectllm_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
detectllm_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:0" if torch.cuda.device_count() > 1 else DEVICE_1


class DetectLLM(object):
    def __init__(self,
                 observer_name_or_path: str = "./Model/falcon-7b",
                 performer_name_or_path: str = "./Model/falcon-7b-instruct",
                 # observer_name_or_path: str = "/data/llm/Llama-3-8B",
                 # performer_name_or_path: str = "/data/llm/Llama-3-8B-Instruct",
                 # observer_name_or_path: str = "/data/llm/Mistral/Mistral-7B-v0.1",
                 # performer_name_or_path: str = "/data/llm/Mistral/Mistral-7B-Instruct-v0.1",
                 # observer_name_or_path: str = "/data/llm/Qwen/Qwen2.5-7B",
                 # performer_name_or_path: str = "/data/llm/Qwen/Qwen2.5-7B-Instruct",
                 # observer_name_or_path: str = "/data/llm/llama-7b",
                 # performer_name_or_path: str = "/data/llm/llama2-7b",
                 use_bfloat16: bool = False,
                 max_token_observed: int = 1024,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   # token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    # token=huggingface_config["TOKEN"]
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = detectllm_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = detectllm_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def cleanup(self):
        if self.observer_model is not None:
            del self.observer_model
            del self.performer_model
            self.observer_model = None
            self.performer_model = None
        torch.cuda.empty_cache()

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        # max_token_indices = performer_logits.argmax(dim=-1)
        # standard_encodings = encodings.copy()
        # standard_encodings.input_ids = max_token_indices

        # standard_ppl = min_perplexity(encodings, performer_logits)

        # standard_x_ppl = entropy_pro(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
        #                 encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        # # standard_detectllm_scores = (standard_ppl / standard_x_ppl).tolist()
        # print("standard:", standard_ppl, standard_x_ppl, standard_ppl/standard_x_ppl)

        # ppl_random = perplexity(encodings, observer_logits)
        # x_ppl_random = entropy(performer_logits.to(DEVICE_1), observer_logits.to(DEVICE_1),
        #                 encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        # print("inverse:", ppl_random, x_ppl_random)

        # ppl = auc_perplexity(encodings, performer_logits, repair_order='l2h')
        ppl = sum_perplexity(encodings, performer_logits)
        # ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        # detectllm_scores = (ppl + standard_ppl) / (x_ppl * 2)
        detectllm_scores = ppl / (2*x_ppl)
        # detectllm_scores = joint_score_ratio(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        detectllm_scores = detectllm_scores.tolist()

        # detectllm_scores = ppl_auc / x_ppl
        # detectllm_scores = detectllm_scores.tolist()

        # relative_ppl = relative_perplexity(encodings, performer_logits)
        # relative_xppl = relative_entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
        #                 encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        # print("相对：", relative_ppl, relative_xppl, relative_ppl/relative_xppl)
        # wppl = weighted_perplexity(encodings, performer_logits)
        # wxppl = w_entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
        #                 encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        # new_scores = wppl/wxppl
        # print("新分数:", wppl, wxppl, new_scores)

        # cos_simi = (ppl*standard_ppl+x_ppl*standard_x_ppl)/(((ppl**2+x_ppl**2)**0.5)*((standard_ppl**2+standard_x_ppl**2)**0.5))
        # print("余弦相似度:", cos_simi)
        # global_value = (ppl - 0.9015 * standard_x_ppl)/(x_ppl - standard_x_ppl)
        # std_local = calculate_window_entropy(observer_logits = observer_logits.to(DEVICE_1), performer_logits=performer_logits.to(DEVICE_1),
        #                 encoding = encodings.to(DEVICE_1), pad_token_id = self.tokenizer.pad_token_id, global_value=((ppl+standard_ppl)/x_ppl))
        #
        # print("global:", global_value)

        print("scores:", ppl, x_ppl, detectllm_scores)
        # print("scores:", ppl+standard_ppl, x_ppl, detectllm_scores)


        return detectllm_scores
        # return ppl.tolist(), x_ppl.tolist(), standard_ppl.tolist(), standard_x_ppl.tolist() if isinstance(input_text, str) else detectllm_scores

    def compute_anomaly_level(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)



        ppl_auc = auc_perplexity(encodings, performer_logits, repair_order="h2l")

        # ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)


        detectllm_scores = ppl_auc / x_ppl
        detectllm_scores = detectllm_scores.tolist()


        print("scores:", ppl_auc, x_ppl, detectllm_scores)

        return detectllm_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        detectllm_scores = np.array(self.compute_score(input_text))
        pred = np.where(detectllm_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
