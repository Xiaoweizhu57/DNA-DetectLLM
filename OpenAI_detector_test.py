import math

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
from transformers import pipeline
from tqdm import tqdm
##Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("/data/zhuxiaowei/project/Model/openAI_detector")
# model = AutoModelForSequenceClassification.from_pretrained("/data/zhuxiaowei/project/Model/openAI_detector")

# Make sure to set the model to evaluation mode
# model.eval()

# text = "\nThis paper addresses the problem of understanding trained CNNs by indexing neuron selectivity. The paper has strengths in that it proposes a new method for indexing neuron selectivity, and that it is able to improve the state-of-the-art in this area. The paper also has weaknesses in that it does not include a comprehensive evaluation of the proposed method, and that it does not consider alternative methods for indexing neuron selectivity."
pipe = pipeline("text-classification", model="/data/zhuxiaowei/project/Model/openAI_detector")
# print(pipe(text))
def openai_detector(text, pipe, tokenizer):
# Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    token_lengths = inputs['input_ids'].shape[1]
    result = pipe(text, truncation=True)
    if result[0]["label"] == 'Real':
        tmp_score = 1 - result[0]["score"]
    elif result[0]["label"] == 'Fake':
        tmp_score = result[0]["score"]
    return tmp_score, token_lengths
    # Perform inference (get the logits)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #
    # Logits = outputs.logits
    # print(Logits)
    # softmax_output = F.softmax(Logits, dim=1)
    # print(softmax_output[0, 1])

    # return torch.argmax(outputs.logits, dim=1).item()
# prob, length = openai_detector(text, model, tokenizer)
# print(length)
# with open("/data/zhuxiaowei/project/AIDetection/Related_dataset/DSB/DSB_calibrate.json", 'r+',
#           encoding='utf-8') as file:
#     data = json.load(file)
#     calibrate_list = data.get("human_text", [])
# #
with open("/data/zhuxiaowei/project/AIDetection/Related_dataset/MAGE/MAGE_human_test_sample1000.json", 'r+',
          encoding='utf-8') as file:
    data = json.load(file)
    human_test_list = data.get("human_text", [])
# # #
# # #
with open("/data/zhuxiaowei/project/AIDetection/Related_dataset/MAGE/MAGE_machine_test_sample1000.json", 'r+',
          encoding='utf-8') as file:
    data = json.load(file)
    machine_test_list = data.get("machine_text", [])
# # # #
# cal_scores = []
# cal_len = []
val_scores = []
val_len = []
test_scores = []
test_len = []
# #
# for cal_text in tqdm(calibrate_list):
#     # preds = openai_detector(cal_text, model, tokenizer)
#     # print(preds)
#     tmp_score, tmp_len = openai_detector(cal_text, pipe, tokenizer)
#     cal_scores.append(tmp_score)
#     cal_len.append(tmp_len)
#
for val_text in tqdm(human_test_list):
    # preds = openai_detector(val_text, model, tokenizer)
    # print(preds)
    tmp_score, tmp_len = openai_detector(val_text, pipe, tokenizer)
    val_scores.append(tmp_score)
    val_len.append(tmp_len)
#
for test_text in tqdm(machine_test_list):
    # preds = openai_detector(test_text, model, tokenizer)
    # print(preds)
    tmp_score, tmp_len = openai_detector(test_text, pipe, tokenizer)
    test_scores.append(tmp_score)
    test_len.append(tmp_len)
# #
# cal_results = {
#             'text': calibrate_list,
#             'predictions': cal_scores,
#             'length': cal_len
#         }
#
val_results = {
            'text': human_test_list,
            'predictions': val_scores,
            'length': val_len
        }

test_results = {
            'text': machine_test_list,
            'predictions': test_scores,
            'length': test_len
        }
# #
with open("/data/zhuxiaowei/project/AIDetection/DNA-DetectLLM/scores/MAGE_openai_detector_human_test.json", 'w') as fout:
    json.dump(val_results, fout)
# #
# with open("/data/zhuxiaowei/project/AIDetection/fast-detect-gpt/scores/DSB_openai_detector_calibrate.json", 'w') as fout:
#     json.dump(cal_results, fout)
#
with open("/data/zhuxiaowei/project/AIDetection/DNA-DetectLLM/scores/MAGE_openai_detector_machine_test.json", 'w') as fout:
    json.dump(test_results, fout)



# with open("/data/zhuxiaowei/project/AIDetection/fast-detect-gpt/scores/RAID_openai_detector_calibrate.json", 'r+',
#           encoding='utf-8') as file:
#     data = json.load(file)
#     score_list = data.get("predictions", [])
#     length_list = data.get("length", [])

# with open("/data/zhuxiaowei/project/AIDetection/DNA-DetectLLM/main_data/xsum_human.json", 'r+',
#           encoding='utf-8') as file:
#     HT_data = json.load(file)
#     HT_score_list = HT_data.get("predictions", [])
#     HT_length_list = HT_data.get("length", [])
#
# with open("/data/zhuxiaowei/project/AIDetection/DNA-DetectLLM/main_data/xsum_machine.json", 'r+',
#           encoding='utf-8') as file:
#     MT_data = json.load(file)
#     MT_score_list = MT_data.get("predictions", [])
#     MT_length_list = MT_data.get("length", [])
#
# ori_TPR_list = []
# ori_F1_list = []
# TPR_list = []
# F1_list = []
# bucket_len = 100
# def normal_analysis(alpha, HT_score_list, MT_score_list):
#     y_true = [0] * len(HT_score_list) + [1] * len(MT_score_list)
#
#     # 合并预测分数
#     y_scores = HT_score_list + MT_score_list
#
#     # 计算FPR, TPR 和 阈值
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     auroc_value = auc(fpr, tpr)
#
#     # print("auroc值为", auroc_value)
#
#     # 查找假阳率为5%时的真阳率
#     target_fpr = alpha
#     closest_index = np.argmin(np.abs(fpr - target_fpr))
#     tpr_at_5_percent_fpr = tpr[closest_index]
#     threshold_at_fpr = thresholds[closest_index]
#
#     # Step 3: 根据该阈值将预测分数转为二进制标签
#     y_pred = np.where(y_scores >= threshold_at_fpr, 1, 0)
#
#     # Step 4: 计算该阈值下的 F1 值
#     f1 = f1_score(y_true, y_pred)
#
#     print(f"Roberta 真阳率 (TPR) 在假阳率 (FPR) 为 {alpha} 时: {tpr_at_5_percent_fpr}")
#     print(f"Roberta F1 值在假阳率 (FPR) 为 {target_fpr} 时: {f1}")
#     ori_F1_list.append(f1)
#     ori_TPR_list.append(tpr_at_5_percent_fpr)
#
# def CP_analysis(alpha, cal_scores, HT_score_list, MT_score_list):
#     y_true = [0] * len(HT_score_list) + [1] * len(MT_score_list)
#     n = len(cal_scores)
#     qhat = np.quantile(cal_scores, max(0, min(1, np.ceil((n + 1) * (1 - alpha)) / n)))
#     val = HT_score_list > qhat
#     test = MT_score_list > qhat
#     normal_pred = np.concatenate((val, test)).tolist()
#     f1 = f1_score(y_true, normal_pred)
#     print("共形预测普通框架下表现 FPR: ", val.mean(),"TPR:", test.mean(), "f1:", f1)
#
#     # print("TPR:", test.mean())
#
# section_len = math.ceil(500/bucket_len)
# # section_len = math.ceil(500/bucket_len) + 1
# cal_total = []
# cal_scores = [[] for _ in range(section_len)]
# for i in range(len(score_list)):
#     cal_total.append(score_list[i])
#     # if length_list[i] >= 500:
#     if length_list[i] >= 500 - bucket_len:
#         cal_scores[section_len - 1].append(score_list[i])
#     else:
#         tmp = length_list[i] // bucket_len
#         cal_scores[tmp].append(score_list[i])
#
# # alpha = 0.2
# for alpha in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
#     qhat_list = []
#     human_calibrated_scores = []
#     machine_calibrated_scores = []
#     for i in range(0, section_len):
#         n = len(cal_scores[i])
#         qhat = np.quantile(cal_scores[i], max(0, min(1, np.ceil((n + 1) * (1 - alpha)) / n)))
#         qhat_list.append(qhat)
#
#     val_label = []
#     test_label = []
#     for i in range(len(HT_score_list)):
#         ##human_result
#         tmp_len = HT_length_list[i]
#         # if tmp_len >= 500:
#         if tmp_len >= 500 - bucket_len:
#             tmp_qhat = qhat_list[section_len - 1]
#         else:
#             tmp_qhat = qhat_list[tmp_len // bucket_len]
#         human_result = HT_score_list[i] > tmp_qhat
#         human_calibrated_scores.append(HT_score_list[i] - tmp_qhat)
#         val_label.append(human_result)
#
#         ##machine_result
#         tmp_len = MT_length_list[i]
#         # if tmp_len >= 500:
#         if tmp_len >= 500 - bucket_len:
#             tmp_qhat = qhat_list[section_len - 1]
#         else:
#             tmp_qhat = qhat_list[tmp_len // bucket_len]
#         machine_result = MT_score_list[i] > tmp_qhat
#         machine_calibrated_scores.append(MT_score_list[i] - tmp_qhat)
#         test_label.append(machine_result)
#     print("FPR:", np.mean(val_label))
#     print("TPR", np.mean(test_label))
#     normal_analysis(alpha, HT_score_list, MT_score_list)
#     CP_analysis(alpha, cal_total, HT_score_list, MT_score_list)
#     TPR_list.append(np.mean(test_label))
#     y_true = [0] * len(HT_score_list) + [1] * len(MT_score_list)
#     y_calibrated_scores = human_calibrated_scores + machine_calibrated_scores
#     fpr, tpr, threshold = roc_curve(y_true, y_calibrated_scores)
#     auroc_value = auc(fpr, tpr)
#     # print("校准后的auroc值为", auroc_value)
#     y_pred_caled = val_label + test_label
#     f1 = f1_score(y_true, y_pred_caled)
#     print("校准后的F1值为", f1)
#     F1_list.append(f1)
# vanilla = []
# RDF = []
# for i in range(len(TPR_list)):
#     vanilla.append(ori_TPR_list[i])
#     vanilla.append(ori_F1_list[i])
#     RDF.append(TPR_list[i])
#     RDF.append(F1_list[i])
# print(vanilla)
# print(RDF)