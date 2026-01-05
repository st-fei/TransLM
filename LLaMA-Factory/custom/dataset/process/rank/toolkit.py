'''
@description 工具包
@date 2025-11-28
'''
import os
import json
import pandas as pd
from typing import Optional

def load_json(json_path):
    '''加载JSON数据'''
    assert os.path.exists(json_path), f"{json_path}不存在"
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def load_jsonl(jsonl_path):
    '''加载JSONL数据'''
    data = []
    # 以UTF-8编码打开文件，with语句自动管理文件句柄
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除首尾空白（含换行符），跳过空行
            clean_line = line.strip()
            if clean_line:
                # 解析单行JSON并加入列表
                data.append(json.loads(clean_line))
    return data

def is_invalid(text):
    '''判断提取到的信息是否无效'''
    stop_words = [0, "0", "无", "None", None]
    
    # 检查是否为NaN值（包括numpy.nan和float('nan')）
    if pd.isna(text):
        return True
    
    return text in stop_words

def build_sampling_dict(
    split: str,
    sampling_mode: str = "random",
    resampling_num: int = 1,
    teacher_ranking_data: Optional[dict] = None,
):
    '''
    若topK>1，则需要为当前样本进行rank采样，构建sampling_dict
    # split:
        # 数据集划分，train/dev/test
        # 用于区分split，从而确定当前是否为train样本，是否要进行重采样
    # sampling_mode:
        # - random mode：只需要令top1为正确答案，后续排序随意构造
        # - teacher mode：以原本模型rank的输出为混淆项，接在top1后面
    # resampling_num:
        # - 大于1：对每个训练样本进行多次采样，每次采样的数量为resampling_num
        # - 等于1：对每个训练样本仅进行一次采样
    # teacher_ranking_data:
        # - 当前这条样本对应teacher模型rank输出
        # - 只需要train/dev的版本即可，test直接计算指标
    '''
    
    return {
        "split": split, 
        "sampling_mode": sampling_mode,
        "resampling_num": resampling_num,
        "teacher_ranking_data": teacher_ranking_data,
    }
