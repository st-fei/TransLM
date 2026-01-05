'''
@description 负责将Sticker数据集转换为LLaMA Factory所需的格式
@date 2025-11-28
'''

import os
import json
import random
import datetime
import argparse
from tqdm import tqdm
from toolkit import load_json, build_sampling_dict
from prompt import PromptManager
from sticker_loader import StickerDataset

parser = argparse.ArgumentParser(description='StickerDataset->Alpaca or ShareGPT')
# 路径控制
parser.add_argument("--dataset_dir", type=str, default="/pscd_dataset/collected_dataset/raw", help="数据集目录")
parser.add_argument("--meme_dir", type=str, default="/pscd_dataset/meme", help="表情包目录")
parser.add_argument("--blogger_dir", type=str, default="/pscd_dataset/blogger", help="博主目录")
parser.add_argument("--persona_dir", type=str, default="/pscd_dataset/persona", help="个性化目录")
parser.add_argument("--output_dir", type=str, default="/pscd_dataset/store/alpaca/rank", help="输出目录")
parser.add_argument("--prompt_path", type=str, default="/pscd_dataset/prompts/prompt.json", help="prompt模版路径")
parser.add_argument("--anno_mapping_path", type=str, default="/pscd_dataset/annotation/mapping.json", help="标注文件映射路径")
parser.add_argument("--blogger_caption_path", type=str, help="发帖图像描述路径")
parser.add_argument("--teacher_ranking_train_path", type=str, help="teacher模型rank输出路径（训练集）")
parser.add_argument("--teacher_ranking_dev_path", type=str, help="teacher模型rank输出路径（验证集）")

# 模式选择
parser.add_argument("--anno_key", type=str, default="qwen3-vl-2b-caption", help="选取的标注模型在mapping.json中的key")
parser.add_argument("--data_mode", type=str, default="random", help="数据集模式")
parser.add_argument("--topK", type=int, default=1, help="TopK")
parser.add_argument("--prompt_mode", type=int, default=0, help="每个rank任务对应多种prompt。-1:为随机选择|0-4为固定选择")
parser.add_argument("--transfer_mode", type=str, default="alpaca", choices=["alpaca", "sharegpt"], help="转换为LLaMA Factory所支持的哪种格式")
parser.add_argument("--type", type=str, default="sft", choices=["sft", "dpo", "s-dpo"], help="转换训练类型")
parser.add_argument("--sampling_mode", type=str, default="random", choices=["random", "teacher"], help="采样模式")
parser.add_argument("--resampling_num", type=int, default=1, help="resampling_num=1: 单训练样本采样，resampling_num>1: 对每个训练样本进行多次采样")
parser.add_argument("--use_blogger_caption", type=int, default=1, help="是否添加blogger caption")

# 参数控制
parser.add_argument("--context_elems", type=str, default="title-desc-tags", help="上下文元素")
parser.add_argument("--max_title_length", type=int, default=20, help="最大发帖标题长度")
parser.add_argument("--max_desc_length", type=int, default=60, help="最大发帖描述长度")
parser.add_argument("--max_tags_num", type=int, default=3, help="最大标签数量")
parser.add_argument("--max_caption_length", type=int, default=60, help="最大发帖图像描述长度")
parser.add_argument("--max_meme_desc_length", type=int, default=60, help="最大表情包描述长度")
parser.add_argument("--max_meme_ocr_length", type=int, default=40, help="最大表情包OCR长度")
parser.add_argument("--only_meme_desc", type=bool, default=True, help="仅使用表情包描述")
parser.add_argument("--seed", type=int, default=42, help="随机种子")

args = parser.parse_args()

def main(args):

    # 设置随机种子
    random.seed(args.seed)

    # 捕获当前时间
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 初始化PromptManager
    prompt_manager = PromptManager(
        prompt_path=args.prompt_path,
        anno_mapping_path=args.anno_mapping_path,
        anno_key=args.anno_key,
        context_elems=args.context_elems,
        max_title_length=args.max_title_length,
        max_desc_length=args.max_desc_length,
        max_tags_num=args.max_tags_num,
        max_caption_length=args.max_caption_length,
        max_meme_desc_length=args.max_meme_desc_length,
        max_meme_ocr_length=args.max_meme_ocr_length,
        only_meme_desc=args.only_meme_desc,
        seed=args.seed,
    )
    # 按split进行处理
    for split in ["train", "dev", "test"]:

        # 创建输出文件路径
        output_dir = os.path.join(args.output_dir, args.transfer_mode, "rank", args.anno_key, current_time)
        os.makedirs(output_dir, exist_ok=True)
        if args.topK > 1:
            if args.sampling_mode == "teacher":
                teacher_model = args.teacher_ranking_train_path.split("/")[-1].split("_")[0]
                if args.use_blogger_caption:
                    output_path = os.path.join(output_dir, f"BloggerCaption_stickerTop_{args.topK}-Sampling-{args.sampling_mode}_{teacher_model}-ResamplingNum-{args.resampling_num}-Use_{args.prompt_mode}-{split}_{args.type}.json")
                else:
                    output_path = os.path.join(output_dir, f"stickerTop_{args.topK}-Sampling-{args.sampling_mode}_{teacher_model}-ResamplingNum-{args.resampling_num}-Use_{args.prompt_mode}-{split}_{args.type}.json")
            else:
                if args.use_blogger_caption:
                    output_path = os.path.join(output_dir, f"BloggerCaption_stickerTop_{args.topK}-Sampling-{args.sampling_mode}-ResamplingNum-{args.resampling_num}-Use_{args.prompt_mode}-{split}_{args.type}.json")
                else:
                    output_path = os.path.join(output_dir, f"stickerTop_{args.topK}-Sampling-{args.sampling_mode}-ResamplingNum-{args.resampling_num}-Use_{args.prompt_mode}-{split}_{args.type}.json")
        else:
            output_path = os.path.join(output_dir, f"stickerTop_{args.topK}-Use_{args.prompt_mode}-{split}_{args.type}.json")
        # 存储处理结果
        data = []
        
        # 加载数据集
        dataset = StickerDataset(
            dataset_dir=args.dataset_dir,
            meme_dir=args.meme_dir,
            blogger_dir=args.blogger_dir,
            persona_dir=args.persona_dir,
            split=split,
            data_mode=args.data_mode,
            topk=args.topK,
            blogger_caption_path=args.blogger_caption_path,
            use_blogger_caption=args.use_blogger_caption
        )

        # 加载teacher_ranking_data
        teacher_ranking_data = None
        # check
        if args.sampling_mode == "teacher" and split == "train" and args.teacher_ranking_train_path is None:
            raise ValueError("教师模式下需要提供teacher_ranking_train_path")
        if args.sampling_mode == "teacher" and split == "dev" and args.teacher_ranking_dev_path is None:
            raise ValueError("教师模式下需要提供teacher_ranking_dev_path")
        # load
        if args.sampling_mode == "teacher" and split == "train" and args.teacher_ranking_train_path is not None:
            teacher_ranking_data = load_json(args.teacher_ranking_train_path)
        elif args.sampling_mode == "teacher" and split == "dev" and args.teacher_ranking_dev_path is not None:
            teacher_ranking_data = load_json(args.teacher_ranking_dev_path)

        # 遍历数据集，将每个样本转换为LLaMA Factory所需的sft格式
        for i, sample in tqdm(enumerate(dataset), desc=f"Processing {split} set Now !"):
            if args.transfer_mode == "alpaca":
                # 转换为Alpaca格式
                # <system>, <instruction>, <input>, <output>, <history>

                # 为当前样本构造sampling_dict，用于topK>1时的采样处理
                sampling_dict = None
                if args.topK > 1:
                    sampling_dict = build_sampling_dict(
                        split=split,
                        sampling_mode=args.sampling_mode,
                        resampling_num=args.resampling_num,
                        teacher_ranking_data=teacher_ranking_data[i] if isinstance(teacher_ranking_data, list) else None,
                    )
                # 若split=train and resampling_num>1，返回的是list[dic]
                maybe_dic_or_list = prompt_manager.shape_alpaca_prompt(
                    topK=args.topK,
                    prompt_mode=args.prompt_mode,
                    type=args.type,
                    sample=sample,
                    sampling_dict=sampling_dict, # 当前样本的sampling_dict
                )
            elif args.transfer_mode == "sharegpt":
                # 转换为ShareGPT格式
                pass

            # 加入data
            if isinstance(maybe_dic_or_list, list):
                data.extend(maybe_dic_or_list)
            else:
                data.append(maybe_dic_or_list)

        # 写入output_path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # 将当前参数记录到output_dir
    with open(os.path.join(output_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main(args)