eval_script_path=/pyproject/sticker_chat/method/LLaMA-Factory/custom/scripts/predict/qwen3_4b_rank10.yaml
CUDA_VISIBLE_DEVICES=1 lmf train ${eval_script_path}

# 多卡
# lmf train ${eval_script_path}