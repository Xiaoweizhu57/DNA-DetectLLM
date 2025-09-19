import numpy as np
import torch
import transformers
import random
from torch.nn import functional as F
from scipy.spatial.distance import jensenshannon
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)




def min_perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0
               ):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    # print("shifted_attention_mask:", shifted_attention_mask)
    # print("shifted_logits:", shifted_logits)
    # Maximize token probability at each time step
    max_prob_token = torch.argmax(shifted_logits,
                                  dim=-1)  # This picks the token with the max probability at each position

    # probs = F.log_softmax(shifted_logits, dim=-1)
    # print("probs:", probs)
    # print("max_prob_token:",max_prob_token)
    # Create a shifted labels tensor for the original sequence
    shifted_labels_max = max_prob_token
    # max_vals = probs.max(dim=-1, keepdim=True)[0]
    # max_vals = (probs - max_vals + 1e-12).max(dim=-1, keepdim=True)[0]
    # print("max_val:", max_vals)
    # print(shifted_labels_max.size())
    # Now, for the perplexity calculation, we compare the shifted logits with the max token sequence
    if median:
        # Compute the cross-entropy loss based on max token sequence
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max)
                  .masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        # Compute perplexity using the max token sequence and attention mask
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        # ppl_test = (F.nll_loss(probs.transpose(1, 2), shifted_labels_max) *
        #        shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        # print("test ppl:", ppl_test)
        ppl = ppl.to("cpu").float().numpy()
        # print("ppl_size:", ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max).size())

    return ppl

def auc_perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0,
               max_batch_size: int = 50,
               repair_order: str="s"):
    shifted_logits = logits[..., :-1, :] / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:]

    probs = torch.softmax(shifted_logits, dim=-1)
    max_prob_tokens = probs.argmax(dim=-1)

    input_ids = encoding.input_ids[..., 1:].clone()

    # 初始的labels是原始真实labels
    current_labels = input_ids.clone()

    # 记录初始状态PPL
    ce_initial = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels).float()
    ppl_sequence = [(ce_initial * shifted_attention_mask).sum() / shifted_attention_mask.sum()]

    logits_diff = torch.abs(probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1) - probs.gather(-1, max_prob_tokens.unsqueeze(-1)).squeeze(-1))

    # 找到非最大概率位置
    non_max_mask = (input_ids != max_prob_tokens)
    change_indices = non_max_mask.nonzero(as_tuple=False)
    if repair_order == "s":
        sorted_indices = change_indices
    elif repair_order == "h2l":
        sorted_indices = sorted(change_indices.tolist(), key=lambda idx: logits_diff[idx[0], idx[1]].item())
    elif repair_order == "l2h":
        sorted_indices = sorted(change_indices.tolist(), key=lambda idx: -logits_diff[idx[0], idx[1]].item())
    elif repair_order == "r":
        sorted_indices = change_indices.tolist()
        random.shuffle(sorted_indices)
    # 逐步用最大概率token替换真实labels
    for idx in change_indices:
        batch_idx, token_idx = idx
        current_labels[batch_idx, token_idx] = max_prob_tokens[batch_idx, token_idx]

        # 使用固定logits和修复后的labels重新计算PPL
        ce_current = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels)
        current_ppl = (ce_current * shifted_attention_mask).sum() / shifted_attention_mask.sum()
        current_ppl = current_ppl.float()
        ppl_sequence.append(current_ppl)

    # 转换为numpy计算AUC
    ppl_sequence_np = torch.stack(ppl_sequence).cpu().numpy()
    auc = np.mean(ppl_sequence_np)

    # print("PPL序列:", ppl_sequence_np)
    # print("AUC:", auc)

    # shifted_logits = (logits[..., :-1, :] / temperature).float()
    # shifted_attention_mask = encoding.attention_mask[..., 1:].float()
    # input_ids = encoding.input_ids[..., 1:].clone()
    #
    # probs = torch.softmax(shifted_logits, dim=-1)
    # max_prob_tokens = probs.argmax(dim=-1)
    #
    # non_max_mask = (input_ids != max_prob_tokens)
    # batch_idx, token_idx = non_max_mask.nonzero(as_tuple=True)
    # total_steps = len(batch_idx)
    #
    # ppl_sequence = []
    # # ppl_sequence.append(perplexity(encoding, logits))
    #
    # current_labels = input_ids.clone()
    # batch_labels = [current_labels.clone()]
    # with torch.no_grad():
    #     for start in range(0, total_steps, max_batch_size):
    #         end = min(start + max_batch_size, total_steps)
    #
    #         for i in range(start, end):
    #             current_labels[batch_idx[i], token_idx[i]] = max_prob_tokens[batch_idx[i], token_idx[i]]
    #             batch_labels.append(current_labels.clone())
    #
    #         partial_batch_labels = torch.stack(batch_labels[-(end - start + 1):])
    #
    #         steps_plus_1, batch, seq_len = partial_batch_labels.size()
    #
    #         logits_expanded = shifted_logits.unsqueeze(0).repeat(steps_plus_1, 1, 1, 1)
    #         mask_expanded = shifted_attention_mask.unsqueeze(0).repeat(steps_plus_1, 1, 1)
    #
    #         logits_flat = logits_expanded.reshape(-1, logits_expanded.size(-1))
    #         labels_flat = partial_batch_labels.reshape(-1)
    #
    #         ce_loss = ce_loss_fn(logits_flat, labels_flat).float()
    #         ce_loss = ce_loss.view(steps_plus_1, batch, seq_len)
    #
    #         ppl_partial = (ce_loss * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2)
    #         if end-start >= 50:
    #             ppl_partial = ppl_partial[:len(ppl_partial)-1]
    #         print(len(ppl_partial))
    #         ppl_sequence.extend(ppl_partial.mean(dim=1).cpu().numpy().tolist())

    # shifted_logits = (logits[..., :-1, :] / temperature).float()
    # shifted_attention_mask = encoding.attention_mask[..., 1:].float()
    # input_ids = encoding.input_ids[..., 1:].clone()
    #
    # probs = torch.softmax(shifted_logits, dim=-1)
    # max_prob_tokens = probs.argmax(dim=-1)
    # non_max_mask = (input_ids != max_prob_tokens)
    #
    # # === 计算 token 差异度（用于排序） ===
    # probs_gt = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    # probs_max = probs.gather(-1, max_prob_tokens.unsqueeze(-1)).squeeze(-1)
    # token_diff = torch.abs(probs_gt - probs_max)
    #
    # batch_idx, token_idx = non_max_mask.nonzero(as_tuple=True)
    #
    # # === 计算替换顺序 ===
    # if repair_order == 'l2h':
    #     sort_keys = token_diff[batch_idx, token_idx]
    #     sorted_indices = torch.argsort(-sort_keys)  # 大到小
    # elif repair_order == 'h2l':
    #     sort_keys = token_diff[batch_idx, token_idx]
    #     sorted_indices = torch.argsort(sort_keys)  # 小到大
    # elif repair_order == 'r':
    #     sorted_indices = torch.randperm(len(batch_idx))
    # else:
    #     sorted_indices = torch.arange(len(batch_idx))  # 默认按原顺序
    #
    # batch_idx = batch_idx[sorted_indices]
    # token_idx = token_idx[sorted_indices]
    # total_steps = len(batch_idx)
    #
    # ppl_sequence = []
    # current_labels = input_ids.clone()
    # batch_labels = [current_labels.clone()]
    #
    # with torch.no_grad():
    #     for start in range(0, total_steps, max_batch_size):
    #         end = min(start + max_batch_size, total_steps)
    #
    #         for i in range(start, end):
    #             current_labels[batch_idx[i], token_idx[i]] = max_prob_tokens[batch_idx[i], token_idx[i]]
    #             batch_labels.append(current_labels.clone())
    #
    #         partial_batch_labels = torch.stack(batch_labels[-(end - start + 1):])  # (steps+1, B, T)
    #         steps_plus_1, batch, seq_len = partial_batch_labels.size()
    #
    #         logits_expanded = shifted_logits.unsqueeze(0).expand(steps_plus_1, -1, -1, -1)
    #         mask_expanded = shifted_attention_mask.unsqueeze(0).expand(steps_plus_1, -1, -1)
    #
    #         logits_flat = logits_expanded.reshape(-1, logits_expanded.size(-1))
    #         labels_flat = partial_batch_labels.reshape(-1)
    #
    #         ce_loss = ce_loss_fn(logits_flat, labels_flat).float()
    #         ce_loss = ce_loss.view(steps_plus_1, batch, seq_len)
    #
    #         ppl_partial = (ce_loss * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2)
    #         if end - start >= 50:
    #             ppl_partial = ppl_partial[:len(ppl_partial) - 1]  # 去掉冗余复制的末尾
    #         ppl_sequence.extend(ppl_partial.mean(dim=1).cpu().numpy().tolist())
    #
    # auc = np.mean(ppl_sequence)
    return auc


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    # print("shifted_labels:", shifted_labels)
    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()


    return ppl

import torch
import numpy as np
from torch.nn import CrossEntropyLoss

ce_loss_fn = CrossEntropyLoss(reduction='none')
@torch.no_grad()
def sum_perplexity(encoding: transformers.BatchEncoding,
                   logits: torch.Tensor,
                   median: bool = False,
                   temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :] / temperature
    attention = encoding.attention_mask[..., 1:]
    labels_std = encoding.input_ids[..., 1:]
    labels_max = torch.argmax(shifted_logits, dim=-1)

    logits_T = shifted_logits.transpose(1, 2)

    ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
    ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

    attn_sum = attention.sum(dim=1).clamp(min=1)
    ppl_std = (ce_std * attention).sum(dim=1) / attn_sum
    ppl_max = (ce_max * attention).sum(dim=1) / attn_sum

    return (ppl_std + ppl_max).cpu().numpy()  # ← 保留 tensor，不转 numpy

import torch
import torch.nn.functional as F
import numpy as np

def joint_score_ratio(p_logits: torch.Tensor,
                      q_logits: torch.Tensor,
                      encoding: transformers.BatchEncoding,
                      pad_token_id: int,
                      temperature: float = 1.0,
                      median: bool = False):
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask
    vocab_size = p_logits.shape[-1]

    # --- Cross Entropy Part ---
    p_log_probs = F.log_softmax(p_logits / temperature, dim=-1)
    q_probs = F.softmax(q_logits / temperature, dim=-1)

    kl = F.kl_div(p_log_probs, q_probs, reduction='none').sum(-1)  # [B, L]
    pad_mask = (input_ids != pad_token_id).float()

    if median:
        kl_masked = kl.masked_fill(pad_mask == 0, float('nan'))
        agg_ce = torch.nanmedian(kl_masked, dim=1).values.cpu().numpy()
    else:
        agg_ce = ((kl * pad_mask).sum(dim=1) / pad_mask.sum(dim=1)).cpu().numpy()

    # --- Perplexity Part ---
    shifted_logits = q_logits[:, :-1, :] / temperature
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention = attention_mask[:, 1:]

    logits_T = shifted_logits.transpose(1, 2)
    labels_std = shifted_input_ids
    labels_max = torch.argmax(shifted_logits, dim=-1)

    ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
    ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

    attn_sum = shifted_attention.sum(dim=1).clamp(min=1)
    ppl_std = (ce_std * shifted_attention).sum(dim=1) / attn_sum
    ppl_max = (ce_max * shifted_attention).sum(dim=1) / attn_sum

    return ((ppl_std + ppl_max) / torch.tensor(agg_ce, device=ppl_std.device)).cpu().numpy()


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    # print("padding_mask", padding_mask)
    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce

def relative_entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    max_vals = p_proba.max(dim=-1, keepdim=True)[0]
    p_proba = p_proba/max_vals


    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def entropy_pro(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]

    # Apply temperature scaling
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    # Calculate probabilities by applying softmax
    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    # print(p_proba)
    if sample_p:
        # Sample from p's distribution
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    # Get the token with the maximum probability at each time step
    max_p_token = torch.argmax(p_proba, dim=-1)  # Select the token with maximum probability
    # print(max_p_token)
    q_scores = q_scores.view(-1, vocab_size)

    # Cross-entropy calculation between q and p, based on the max token at each position
    ce = ce_loss_fn(input=q_scores, target=max_p_token).view(-1, total_tokens_available)

    # Create a padding mask to ignore padding tokens
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    # Aggregate cross-entropy loss either by median or mean based on the flag
    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce




def w_entropy(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        encoding: transformers.BatchEncoding,
        pad_token_id: int,
        median: bool = False,
        sample_p: bool = False,
        temperature: float = 1.0
):
    use_weight = True
    shifted_labels = encoding.input_ids[..., :].contiguous()
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)

    # ==== 权重计算（仅在 use_weight=True 时生效）====
    if use_weight:
        # 计算模型 P 的概率分布
        p_probs = F.softmax(p_logits / temperature, dim=-1)  # [batch, seq_len, vocab]

        # 获取目标 token 对应的概率
        current_probs = p_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1)  # [batch, seq_len, 1]
        ).squeeze(-1)  # [batch, seq_len]

        # 计算最大概率差权重
        max_probs = p_probs.max(dim=-1).values  # [batch, seq_len]
        weights = max_probs - current_probs  # [batch, seq_len]
        # 应用权重
        ce = ce * weights

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())



    return agg_ce


def calculate_window_ppl(observer_logits: torch.Tensor,
               performer_logits: torch.Tensor,
               encoding: transformers.BatchEncoding,
               pad_token_id: int,
               global_value: float,
               window_radio: float = 0.1):
    seq_len = len(encoding.input_ids[..., 0:][0])
    ppl_values = []
    xppl_values = []
    standard_ppl_values = []
    standard_xppl_values = []
    local_values = []
    window_size = int(seq_len * window_radio)
    step_size = int(window_size*0.5)
    if seq_len <= 50:
        window_size = int(seq_len * 0.5)
        step_size = int(window_size * 0.5)

    shifted_logits = performer_logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()

    ppls = torch.softmax(shifted_logits, dim=-1) # [batch, seq_len-1, vocab]
    max_ppls, _ = ppls.max(dim=-1) # [batch, seq_len-1]
    max_ppls = max_ppls[0]
    ppls = ppls.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)[0]
    global_cos_sim = ((ppls*max_ppls).sum(dim=-1)) / (ppls.norm(dim=-1) * max_ppls.norm(dim=-1) + 1e-9)

    if seq_len >= window_size:
        ppls_windows = ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]
        max_ppls_windows = max_ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]

        # 计算窗口内余弦相似度
        dot_product = (ppls_windows * max_ppls_windows).sum(dim=-1)
        ppls_norm = ppls_windows.norm(dim=-1)
        max_ppls_norm = max_ppls_windows.norm(dim=-1)
        cos_similarities = dot_product / (ppls_norm * max_ppls_norm + 1e-9)  # 避免除零

        # 计算均值和标准差
        mean_cos_sim = cos_similarities.mean().item()
        std_cos_sim = cos_similarities.std().item()
    else:
        mean_cos_sim = 0.0  # 若数据长度不足以形成窗口，返回默认值

    return (global_cos_sim-mean_cos_sim)/std_cos_sim



def calculate_window_entropy(observer_logits: torch.Tensor,
                         performer_logits: torch.Tensor,
                         encoding: transformers.BatchEncoding,
                         pad_token_id: int,
                         global_value: float,
                         window_radio: float = 0.1):
    seq_len = len(encoding.input_ids[..., 0:][0])
    ppl_values = []
    xppl_values = []
    standard_ppl_values = []
    standard_xppl_values = []
    local_values = []
    window_size = int(seq_len * window_radio)
    step_size = int(window_size * 0.5)
    if seq_len <= 50:
        window_size = int(seq_len * 0.5)
        step_size = int(window_size * 0.5)
    elif seq_len <= 200:
        window_size = int(seq_len * 0.2)
        step_size = int(window_size * 0.5)

    shifted_logits = performer_logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    q_shifted_logits = observer_logits[..., :-1, :].contiguous()

    ppls = torch.softmax(shifted_logits, dim=-1)  # [batch, seq_len-1, vocab]
    max_ppls, _ = ppls.max(dim=-1)  # [batch, seq_len-1]
    max_ppls = max_ppls[0]
    ppls = ppls.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)[0]

    q_ppls = torch.softmax(q_shifted_logits, dim=-1)  # [batch, seq_len-1, vocab]
    q_max_ppls, _ = q_ppls.max(dim=-1)  # [batch, seq_len-1]
    q_max_ppls = q_max_ppls[0]
    q_ppls = q_ppls.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)[0]
    # global_cos_sim = ((ppls*max_ppls).sum(dim=-1)) / (ppls.norm(dim=-1) * max_ppls.norm(dim=-1) + 1e-9)

    if seq_len >= window_size:
        ppls_windows = ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]
        max_ppls_windows = max_ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]

        q_ppls_windows = q_ppls.unfold(0, window_size, step_size)
        q_max_ppls_windows = q_max_ppls.unfold(0, window_size, step_size)

        # 计算窗口内熵值
        p_diff = -torch.log(ppls_windows + 1e-9) + torch.log(max_ppls_windows + 1e-9)
        q_diff = -torch.log(q_ppls_windows + 1e-9) + torch.log(q_max_ppls_windows + 1e-9)

        diff = (p_diff * q_diff)/(p_diff + q_diff + 1e-9)

        # diff = q_diff
        # print("diff:", diff)
        print(diff.mean(dim=1))
        diff_mean = diff.mean(dim=1)
        diff_mean[diff_mean < 0.30] = 0
        # print(diff_mean.mean())


    else:
        diff = 0.0  # 若数据长度不足以形成窗口，返回默认值

    return diff_mean.mean()


    # for i in range(0, seq_len - window_size + 1, step_len):
    # # 获取当前窗口的 logits 和 encoding
    #
    #     tmp_observer_logits = torch.tensor(observer_logits[:, i:i + window_size, :])
    #
    #     window_logits = torch.tensor(performer_logits[:, i:i + window_size, :])
    #
    #     tmp_encoding = encoding.input_ids[..., i:i + window_size]
    #     tmp_mask = encoding.input_ids[..., i:i + window_size]
    #     window_encoding = {'input_ids': tmp_encoding,
    #                        'attention_mask': tmp_mask}
    #     window_encoding = transformers.BatchEncoding(window_encoding)
    #
    #     window_standard_ppl = min_perplexity(window_encoding, window_logits)  # ppl_function 是你的计算困惑度的函数
    #     # window_standard_xppl = entropy_pro(tmp_observer_logits, window_logits, window_encoding, pad_token_id)
    #     standard_ppl_values.append(window_standard_ppl)
    #     # standard_xppl_values.append(window_standard_xppl)
    #
    #     # 计算当前窗口的 PPL（或 XPPL）
    #     window_ppl = perplexity(window_encoding, window_logits)  # ppl_function 是你的计算困惑度的函数
    #     window_xppl = entropy(tmp_observer_logits, window_logits, window_encoding, pad_token_id)
    #     ppl_values.append(window_ppl)
    #     xppl_values.append(window_xppl)
    #     # print("window_ppl:", ppl_values)
    #
    #
    #
    # # local_values = [(x - 0.9015*z)/(y - z) for x,y,z in zip(ppl_values, xppl_values, standard_xppl_values)]
    # local_values = [(x+z)/y for x, y, z in zip(ppl_values, xppl_values, standard_ppl_values)]
    #
    # # D_values = [x - global_value for x in local_values]
    #
    # # print("D values:", D_values)
    # std_local = np.std(local_values)
    # return std_local