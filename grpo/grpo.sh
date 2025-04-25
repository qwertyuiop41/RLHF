

TIMESTAMP2=$(date +"%Y-%m-%d_%H-%M-%S")

python /HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/grpo/train2.py 2>&1 | tee /HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/grpo/grpo2_${TIMESTAMP2}.log