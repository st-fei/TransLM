'''
@author tfshen
@description 自定义参数
@date 2025-12-01
'''

import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from datasets import DownloadMode

@dataclass
class CustomArguments:
    r"""Arguments we use to customize the training/evaluation process."""

    rank_num: int = field(
        default=1,
        metadata={"help": "Number of model-rank for current dataset"},
    )

    candidate_num: int = field(
        default=10,
        metadata={"help": "Number of candidate items"}
    )

    compute_mAP: bool = field(
        default=False,
        metadata={"help": "Whether to compute mAP during evaluation."},
    )

    dataset_id: str = field(
        default="v1",
        metadata={"help": "dataset id"}
    )