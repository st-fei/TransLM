'''
@description 加载与管理prompt
@date 2025-11-28
'''

import os
import json
import random
from typing import Optional, List, Union
from toolkit import load_json, is_invalid
from utils import parse_meme_with_simple_caption, parse_meme_with_caption, parse_meme_with_keyword, parse_meme_with_query, parse_meme_with_caption_and_keyword, parse_meme_with_caption_and_query

class PromptManager:
    def __init__(
        self, 
        prompt_path: str,
        anno_mapping_path: str,
        anno_key: str,
        context_elems: Optional[str] = "title-desc-tags", # context中包含的元素
        max_title_length: Optional[int] = 20, # 最大标题长度
        max_desc_length: Optional[int] = 60, # 最大发帖内容长度
        max_tags_num: Optional[int] = 3, # 最大标签数量
        max_meme_desc_length: Optional[int] = 40, # 最大表情包描述长度
        max_caption_length: Optional[int] = 60, # 最大发帖图像描述长度
        max_meme_ocr_length: Optional[int] = 40, # 最大表情包ocr文本长度
        only_meme_desc: Optional[bool] = True, # 是否只使用表情包描述
        seed: Optional[int] = 42, # 随机种子
    ):
        self.prompt_path = prompt_path
        self.anno_mapping_path = anno_mapping_path
        self.anno_key = anno_key
        self.context_elems = context_elems
        self.prompts = self.load_prompts() # 加载prompts模版
        self.anno, self.meme_load_key = self.load_anno(anno_mapping_path, anno_key) # 加载指定stickers标注文件
        # 参数管理
        self.max_title_length = max_title_length
        self.max_desc_length = max_desc_length
        self.max_tags_num = max_tags_num
        self.max_caption_length = max_caption_length # 最大发帖图像描述长度
        self.max_meme_desc_length = max_meme_desc_length
        self.max_meme_ocr_length = max_meme_ocr_length
        self.only_meme_desc = only_meme_desc
        # 随机种子
        self.seed = seed
        random.seed(seed)


    def load_prompts(self):
        '''加载prompt模版'''
        with open(self.prompt_path, "r") as f:
            prompts = json.load(f)
        return prompts
    
    def load_anno(self, anno_mapping_path: str, anno_key: str):
        '''加载指定stickers标注文件'''
        anno_mapping = load_json(anno_mapping_path)
        anno_info = anno_mapping[anno_key]
        anno_path = anno_info["path"]
        meme_load_key = anno_info["load_key"]
        if isinstance(anno_path, dict):
            anno = {}
            for elem in ["caption", "keyword", "query"]:
                if elem in anno_path:
                    anno[elem] = load_json(anno_path[elem])
        else:
            anno = load_json(anno_path)
        return anno, meme_load_key

    def parse_meme(self, 
        meme_id: Union[str, int],
        max_meme_desc_length: Optional[int] = 40, # 最大表情包描述长度
        max_meme_ocr_length: Optional[int] = 40, # 最大表情包ocr文本长度
        only_meme_desc: Optional[bool] = True, # 是否只使用description
    ):
        '''解析meme_id对应meme的详情'''
        if isinstance(meme_id, int):
            meme_id = str(meme_id)
        meme_key = f"sticker_{meme_id}"
        # 加载meme caption

        if self.meme_load_key == "simple_caption":
            return parse_meme_with_simple_caption(
                meme_key=meme_key,
                max_meme_desc_length=max_meme_desc_length,
                anno_data=self.anno
            )

        if self.meme_load_key == "caption":
            return parse_meme_with_caption(
                meme_key=meme_key,
                max_meme_desc_length=max_meme_desc_length,
                max_meme_ocr_length=max_meme_ocr_length,
                only_meme_desc=only_meme_desc,
                anno_data=self.anno
            )
        # 加载meme keyword
        if self.meme_load_key == "keyword":
            return parse_meme_with_keyword(
                meme_key=meme_key,
                max_meme_desc_length=max_meme_desc_length,
                anno_data=self.anno
            )
        # 同时加载caption and keyword
        elif self.meme_load_key == "caption_and_keyword":
            return parse_meme_with_caption_and_keyword(
                meme_key=meme_key,
                max_meme_desc_length=max_meme_desc_length,
                max_meme_ocr_length=max_meme_ocr_length,
                only_meme_desc=only_meme_desc,
                caption_anno=self.anno["caption"],
                keyword_anno=self.anno["keyword"]
            )
        # 加载query
        elif self.meme_load_key == "query":
            return parse_meme_with_query(
                meme_key=meme_key,
                max_meme_desc_length=max_meme_desc_length,
                anno_data=self.anno
            )
        # 同时加载caption and query
        elif self.meme_load_key == "caption_and_query":
            return parse_meme_with_caption_and_query(
                meme_key=meme_key,
                max_meme_desc_length=max_meme_desc_length,
                max_meme_ocr_length=max_meme_ocr_length,
                only_meme_desc=only_meme_desc,
                caption_anno=self.anno["caption"],
                query_anno=self.anno["query"]
            )
        
    def build_top1_inout(self, sample: dict):
        '''根据sample中的属性构建input & output'''
        # 提取单条sticker样本中的属性
        attribute = sample["attribute"]
        note_id = sample["note_id"]
        note_dir = sample["note_dir"]
        note_title = sample["note_title"]
        note_desc = sample["note_desc"]
        note_tags = sample["note_tags"]
        comment_content = sample["comment_content"]
        reply_content = sample["reply_content"]
        gth_meme_id = sample["gth_meme_id"]
        candidate_meme_ids = sample["candidate_meme_ids"]
        persona_bio = sample["persona_bio"]
        persona_note_id = sample["persona_note_id"]
        persona_note_dir = sample["persona_note_dir"]
        persona_note_title = sample["persona_note_title"]
        persona_note_desc = sample["persona_note_desc"]
        persona_note_tags = sample["persona_note_tags"]
        # 构建context
        context = "某社交媒体平台一发帖内容如下："
        if "title" in self.context_elems:
            context += "发帖标题：{}，".format(note_title[:self.max_title_length])
        if "desc" in self.context_elems:
            context += "发帖详情：{}，".format(note_desc[:self.max_desc_length])
        if "tags" in self.context_elems:
            context += "发帖标签：{}。".format(note_tags[:self.max_tags_num])
        if attribute == "comment":
            context += f"用户1的评论内容为：{comment_content}。请基于用户1的评论，从候选表情包/图像选项中挑选最符合用户1期望的选项，"
        elif attribute == "reply" and not is_invalid(reply_content):
            context += f"用户2的回复内容为：{reply_content}。请基于用户2的回复，从以下候选表情包/图像选项中挑选最符合用户2期望的选项。"
        elif attribute == "reply" and is_invalid(reply_content):
            context += "用户2想用一张表情包/图像回复该评论。请基于上述用户评论，从以下候选表情包/图像选项中挑选最符合用户2期望的选项。"
        # 构建options
        options = ""
        # 为正确选项随机一个位置
        random_index = random.randint(0, len(candidate_meme_ids))
        gth_option = ""
        cnt = 0
        for meme_id in candidate_meme_ids:
            if cnt == random_index:
                options += f"Option_{cnt+1}: {self.parse_meme(gth_meme_id, self.max_meme_desc_length, self.max_meme_ocr_length, self.only_meme_desc)}"
                # TODO 考虑使用'\n'或','分隔不同选项
                options += "\n"
                gth_option = f"Option_{cnt+1}"
                cnt += 1
            options += f"Option_{cnt+1}: {self.parse_meme(meme_id, self.max_meme_desc_length, self.max_meme_ocr_length, self.only_meme_desc)}"
            # TODO 考虑使用'\n'或','分隔不同选项
            options += "\n"
            cnt += 1
        # 若随机到最后一位，单独处理
        if gth_option == "":
            gth_option = f"Option_{random_index+1}"
            options += f"{gth_option}: {self.parse_meme(gth_meme_id, self.max_meme_desc_length, self.max_meme_ocr_length, self.only_meme_desc)}"
        input = f"{context}\n候选项如下：{options}"
        output = gth_option
        return input, output

    def build_topK_inout(self, sample: dict, topK: int, sampling_dict: dict):
        '''
        根据sample中的属性以及选择的采样模式来构建topK选项的input & output
        
        Args:
            sample (dict): 单条sticker数据集样本
            topK (int): 选择rank_{topK}任务
            sampling_dict (dict): 采样模式对应的参数
                split (str): 当前该条数据对应的数据集划分，train/dev/test
                sampling_mode (str): 采样模式，random/teacher
                resampling_num (int): 重采样次数
                teacher_ranking_data (dict): 当前这条样本对应teacher模型rank输出
        Returns:
            input (str): 构建的input
            output_list (list): 构建的output_list，长度为sampling_dict[resampling_num]
        '''
        # 提取单条sticker样本中的属性
        attribute = sample["attribute"]
        note_id = sample["note_id"]
        note_dir = sample["note_dir"]
        note_title = sample["note_title"]
        note_desc = sample["note_desc"]
        note_tags = sample["note_tags"]
        blogger_caption = sample["blogger_caption"]
        comment_content = sample["comment_content"]
        reply_content = sample["reply_content"]
        gth_meme_id = sample["gth_meme_id"]
        candidate_meme_ids = sample["candidate_meme_ids"]
        persona_bio = sample["persona_bio"]
        persona_note_id = sample["persona_note_id"]
        persona_note_dir = sample["persona_note_dir"]
        persona_note_title = sample["persona_note_title"]
        persona_note_desc = sample["persona_note_desc"]
        persona_note_tags = sample["persona_note_tags"]
        # 构建context
        context = "某社交媒体平台一发帖内容如下："
        if "title" in self.context_elems:
            context += "发帖标题：{}，".format(note_title[:self.max_title_length])
        if "desc" in self.context_elems:
            context += "发帖详情：{}，".format(note_desc[:self.max_desc_length])
        if "tags" in self.context_elems:
            context += "发帖标签：{}。".format(note_tags[:self.max_tags_num])
        if "caption" in self.context_elems and blogger_caption is not None:
            context += f"发帖图像内容总结：{blogger_caption[:self.max_caption_length]}。"
        if attribute == "comment":
            context += f"用户1的评论内容为：{comment_content}。请基于用户1的评论，从以下候选表情包/图像选项中挑选最符合用户1期望的前{topK}个选项，并按照格式输出。"
        elif attribute == "reply" and not is_invalid(reply_content):
            context += f"用户2的回复内容为：{reply_content}。请基于用户2的回复，从以下候选表情包/图像选项中挑选最符合用户2期望的前{topK}个选项，并按照格式输出。"
        elif attribute == "reply" and is_invalid(reply_content):
            context += f"用户2想用一张表情包/图像回复该评论。请基于上述用户评论，从以下候选表情包/图像选项中挑选最符合用户2期望的前{topK}个选项，并按照格式输出。"
        # 构建options
        options = ""
        # 为正确选项随机一个位置
        random_index = random.randint(0, len(candidate_meme_ids))
        gth_option = ""
        cnt = 0
        for meme_id in candidate_meme_ids:
            if cnt == random_index:
                options += f"Option_{cnt+1}: {self.parse_meme(gth_meme_id, self.max_meme_desc_length, self.max_meme_ocr_length, self.only_meme_desc)}"
                # TODO 考虑使用'\n'或','分隔不同选项
                options += "\n"
                gth_option = f"Option_{cnt+1}"
                cnt += 1
            options += f"Option_{cnt+1}: {self.parse_meme(meme_id, self.max_meme_desc_length, self.max_meme_ocr_length, self.only_meme_desc)}"
            # TODO 考虑使用'\n'或','分隔不同选项
            options += "\n"
            cnt += 1
        # 若随机到最后一位，单独处理
        if gth_option == "":
            gth_option = f"Option_{random_index+1}"
            options += f"{gth_option}: {self.parse_meme(gth_meme_id, self.max_meme_desc_length, self.max_meme_ocr_length, self.only_meme_desc)}"
        input = f"{context}\n候选项如下：{options}"

        # 按照sampling_dict来构建output_list
        output_list = []

        # 若为random模式
        #!!! random mode构建思路：确定第一个选项为gth_option，其他选项随机排列
        if sampling_dict["sampling_mode"] == "random":
            if sampling_dict["split"] == "test":
                output_list.append(gth_option)
            else:
                rng = random.Random()
                for _ in range(sampling_dict["resampling_num"]):
                    random_list = [i + 1 for i in range(len(candidate_meme_ids) + 1) if i != random_index]
                    rng.shuffle(random_list)  # 使用独立实例的 shuffle，不影响全局
                    output = [gth_option] + [f"Option_{random_list[i]}" for i in range(topK - 1)]
                    output_list.append(str(output))
        elif sampling_dict["sampling_mode"] == "teacher":
            if sampling_dict["split"] == "test":
                output_list.append(gth_option)
            else:
                teacher_ranking = json.loads(sampling_dict["teacher_ranking_data"]["response"])["sticker_rank"]
                if isinstance(teacher_ranking, str):
                    teacher_ranking = teacher_ranking[1:-1]
                    teacher_ranking = teacher_ranking.split(", ")
                # 建立原Option到最新乱序Option的映射
                # <= random_index + 1：Option_{cnt} -> Option_{cnt-1}，向前移位
                # > random_index + 1：Option_{cnt} -> Option_{cnt}，保持不变
                option_mapping = {"Option_1": gth_option}
                for cnt in range(2, len(candidate_meme_ids) + 2):
                    if cnt <= random_index + 1:
                        option_mapping[f"Option_{cnt}"] = f"Option_{cnt-1}"
                    else:
                        option_mapping[f"Option_{cnt}"] = f"Option_{cnt}"
                reversed_option_mapping = {v: k for k, v in option_mapping.items()} # 最新乱序Option到原Option的映射
                output_list = []
                subsequent_list = []
                # 获取subsequent_list
                for i in range(len(teacher_ranking)):
                    ranking_item = teacher_ranking[i]
                    ranking_id = ranking_item.split("_")[-1]
                    ranking_option = f"Option_{ranking_id}"
                    if option_mapping[ranking_option] == gth_option:
                        continue
                    else:
                        subsequent_list.append(ranking_option)
                rng = random.Random()
                for _ in range(sampling_dict["resampling_num"]):
                    gold_list = [gth_option]
                    rng.shuffle(subsequent_list) # 之后的顺序打乱了！！！
                    for subsequent_item in subsequent_list:
                        gold_list.append(option_mapping[subsequent_item])
                    gold_list = gold_list[:topK]
                    output_list.append(str(gold_list))
            
        return input, output_list

    def shape_alpaca_prompt(
        self, 
        topK: int, # 选择rank_{topK}任务
        prompt_mode: int, # -1代表随机|0-4代表固定选择
        type: str, # 对应sft/dpo/s-dpo任务
        sample: dict, # 单条sticker数据集样本
        sampling_dict: Optional[dict] = None, # 采样模式对应的参数
    ):
        '''返回alpaca格式的prompt'''
        if type == "sft": 
            prompt_templates = self.prompts[f"rank{topK}"]
            num_templates = len(prompt_templates) # 当前rank任务共有num_prompts个模版
            if prompt_mode == -1:
                prompt_template = random.choice(prompt_templates)
            else:
                if prompt_mode < num_templates:
                    prompt_template = prompt_templates[prompt_mode] 
                else:
                    prompt_template = random.choice(prompt_templates)
            # 提取选择的prompt模版中的属性
            system = prompt_template["system"]
            instruction = prompt_template["instruction"]
            input = prompt_template["input"]
            output = prompt_template["output"]
            # topK=1，直接根据sample中的属性构建context和options
            if topK == 1:
                builded_input, builded_output = self.build_top1_inout(sample)
                input = input.replace("[Input]", builded_input)
                output = output.replace("[Output]", builded_output)
                # 构建alpaca格式的prompt
                prompt = {
                    "system": system,
                    "instruction": instruction,
                    "input": input,
                    "output": output
                }
            # topK>1，需要根据sample中的属性以及选择的采样模式来构建context和options
            elif topK > 1:
                builded_input, builded_output_list = self.build_topK_inout(sample, topK, sampling_dict)
                input = input.replace("[Input]", builded_input)
                # 未重采样
                if len(builded_output_list) == 1:
                    output = output.replace("[Output]", builded_output_list[0])
                    # 构建alpaca格式的prompt
                    prompt = {
                        "system": system,
                        "instruction": instruction,
                        "input": input,
                        "output": output
                    }
                # 重采样，分批加入
                else:
                    prompt = []
                    for builded_output in builded_output_list:
                        prompt.append({
                            "system": system,
                            "instruction": instruction,
                            "input": input,
                            "output": builded_output
                        })

        return prompt