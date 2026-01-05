sft_script_paths=(
    /pyproject/sticker_chat/method/LLaMA-Factory/custom/scripts/lora_sft/qwen3_0.6b_rank10.yaml
    /pyproject/sticker_chat/method/LLaMA-Factory/custom/scripts/lora_sft/qwen3_1.7b_rank10.yaml
    /pyproject/sticker_chat/method/LLaMA-Factory/custom/scripts/lora_sft/qwen3_4b_rank10.yaml
)

# 执行模式选择：multi_gpu（多卡）/ single_gpu（单卡）
run_mode="multi_gpu"
# 单卡时指定GPU编号
single_gpu_id=1
# ====================================================

# 遍历所有脚本路径执行训练
for script_path in "${sft_script_paths[@]}"; do
    # 检查脚本文件是否存在
    if [ ! -f "$script_path" ]; then
        echo -e "\033[31m[ERROR] 脚本文件不存在：$script_path\033[0m"
        continue  # 跳过不存在的文件，继续执行下一个
    fi

    echo -e "\n\033[32m[INFO] 开始执行脚本：$script_path\033[0m"
    echo -e "[INFO] 执行时间：$(date +'%Y-%m-%d %H:%M:%S')"

    # 根据执行模式选择命令
    if [ "$run_mode" = "single_gpu" ]; then
        # 单卡执行命令
        CUDA_VISIBLE_DEVICES=$single_gpu_id lmf train "$script_path"
    else
        # 多卡执行命令
        lmf train "$script_path"
    fi

    # 检查命令执行结果
    if [ $? -eq 0 ]; then
        echo -e "\033[32m[SUCCESS] 脚本执行完成：$script_path\033[0m"
    else
        echo -e "\033[31m[FAILED] 脚本执行失败：$script_path\033[0m"
        # 如果需要失败后立即退出，取消下面的注释
        # exit 1
    fi
done

echo -e "\n\033[32m[FINISH] 所有脚本遍历执行完成\033[0m"

