from .metrics import recall_2at1, recall_at_k_new, precision_at_k, MRR, MAP
import numpy as np

def evaluation(pred_scores, true_scores, samples=10):
    '''
    :param pred_scores:     list of scores predicted by model
    :param true_scores:     list of ground truth labels, 1 or 0
    :param samples:         候选量，支持10/20/50三个值
    :return:                包含对应评估指标的字典
    '''
    # 合法性检查：确保samples是支持的取值
    if samples not in [10, 20, 50]:
        raise ValueError(f"Unsupported sample size {samples}, only 10/20/50 are allowed")
    
    # 验证输入长度匹配且能被候选量整除
    if len(pred_scores) != len(true_scores):
        raise ValueError("pred_scores and true_scores must have the same length")
    if len(pred_scores) % samples != 0:
        raise ValueError(f"Total length of scores must be divisible by sample size {samples}")
    
    # 拆分样本（适配动态候选量）
    num_sample = len(pred_scores) // samples
    score_list = np.split(np.array(pred_scores), num_sample, axis=0)  # 保留原拆分逻辑
    
    # 计算基础指标（所有候选量都需要的指标）
    recall_2_1 = recall_2at1(np.array(true_scores), np.array(pred_scores))
    recall_at_1 = recall_at_k_new(labels=np.array(true_scores), scores=np.array(pred_scores), k=1, doc_num=samples)
    recall_at_2 = recall_at_k_new(labels=np.array(true_scores), scores=np.array(pred_scores), k=2, doc_num=samples)
    recall_at_5 = recall_at_k_new(labels=np.array(true_scores), scores=np.array(pred_scores), k=5, doc_num=samples)
    _mrr = MRR(np.array(true_scores), np.array(pred_scores), k=samples//2)
    _map = MAP(np.array(true_scores), np.array(pred_scores), k=samples//2)
    precision_at_1 = precision_at_k(np.array(true_scores), np.array(pred_scores), k=1)
    
    # 构建基础结果字典
    result = {
        'MAP': _map,
        'MRR': _mrr,
        'p@1': precision_at_1,
        'r2@1': recall_2_1,
        'r@1': recall_at_1,
        'r@2': recall_at_2,
        'r@5': recall_at_5,
    }
    
    # 根据候选量添加额外的召回率指标
    if samples == 20:
        # 20个候选量：增加recall@10
        recall_at_10 = recall_at_k_new(labels=np.array(true_scores), scores=np.array(pred_scores), k=10, doc_num=samples)
        result['r@10'] = recall_at_10
    elif samples == 50:
        # 50个候选量：增加recall@25
        recall_at_25 = recall_at_k_new(labels=np.array(true_scores), scores=np.array(pred_scores), k=25, doc_num=samples)
        result['r@25'] = recall_at_25
    
    return result
