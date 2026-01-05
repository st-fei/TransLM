'''
@description 加载Sticker数据集
@date 2025-11-28
@extra 先加载Sticker数据集，然后将其转换为LLaMA Factory所需的格式
'''

import os
import json
import torch
import random
from toolkit import load_json
from torch.utils.data import Dataset, DataLoader

class StickerDataset(Dataset):
    def __init__(
        self, 
        dataset_dir,
        meme_dir,
        blogger_dir,
        persona_dir,
        tokenizer=None,
        split="train",
        data_mode="random",
        topk=5,
        blogger_caption_path=None,
        use_blogger_caption=False
    ):
        # 存至实例属性
        self.dataset_dir = dataset_dir
        self.meme_dir = meme_dir
        self.blogger_dir = blogger_dir
        self.persona_dir = persona_dir
        self.tokenizer = tokenizer
        self.split = split
        self.data_mode = data_mode
        self.topk = topk
        self.blogger_caption_path = blogger_caption_path
        self.use_blogger_caption = use_blogger_caption
        self.blogger_caption = load_json(self.blogger_caption_path) if self.use_blogger_caption else None
        # 拼接dataset_path
        if topk == 5:
            self.dataset_path = os.path.join(self.dataset_dir, f"{self.split}_{self.data_mode}.json")
        else:
            self.dataset_path = os.path.join(self.dataset_dir, f"{self.split}_{self.data_mode}_{topk * 2}.json")
        # 加载数据集
        self.data = load_json(self.dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        attribute = item["attribute"]
        note_info = item["note_info"]
        # 发帖信息
        note_id = note_info["note_id"]
        note_title = note_info["title"]
        note_desc = note_info["description"]
        note_tags = note_info["tags"]
        # 发帖图像描述
        blogger_caption_item = self.blogger_caption.get(note_id, None) if self.use_blogger_caption else None
        blogger_caption = blogger_caption_item.get("response", None) if blogger_caption_item is not None else None
        note_dir = os.path.join(self.blogger_dir, note_id)
        # 互动内容
        comment_content = str(item.get("comment_content", "无"))
        reply_content = str(item.get("reply_content", "无")) 
        comment_meme_id = item.get("comment_meme_idx", None)
        reply_meme_id = item.get("reply_meme_idx", None)
        gth_meme_id = item["comment_meme_idx"] if attribute == "comment" else item["reply_meme_idx"]
        candidate_meme_ids = item["comment_candidate_meme_list"] if attribute == "comment" else item["reply_candidate_meme_list"]
        # 个性化信息
        persona_info = item["commenter_persona"] if attribute == "comment" else item["replyer_persona"]
        persona_bio = None
        persona_note_id = None
        persona_note_dir = None
        persona_note_title = None 
        persona_note_desc = None
        persona_note_tags = None
        if isinstance(persona_info, dict):
            persona_bio = persona_info.get("bio", None)
            persona_note_id = persona_info.get("note_info", {}).get("note_id", None)
            if persona_note_id is not None:
                persona_note_dir = os.path.join(self.persona_dir, persona_note_id)
            persona_note_title = persona_info.get("note_info", {}).get("title", None)
            persona_note_desc = persona_info.get("note_info", {}).get("description", None)
            persona_note_tags = persona_info.get("note_info", {}).get("tags", None)
        # 返回数据项
        return {
            "attribute": attribute,
            "note_id": note_id,
            "note_dir": note_dir,
            "note_title": note_title,
            "note_desc": note_desc,
            "note_tags": note_tags,
            "blogger_caption": blogger_caption,
            "comment_content": comment_content,
            "reply_content": reply_content,
            "comment_meme_id": comment_meme_id,
            "reply_meme_id": reply_meme_id,
            "gth_meme_id": gth_meme_id,
            "candidate_meme_ids": candidate_meme_ids,
            "persona_bio": persona_bio,
            "persona_note_id": persona_note_id,
            "persona_note_dir": persona_note_dir,
            "persona_note_title": persona_note_title,
            "persona_note_desc": persona_note_desc,
            "persona_note_tags": persona_note_tags,
        }

