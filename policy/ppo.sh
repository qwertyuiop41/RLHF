TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

python /HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/policy/train.py 2>&1 | tee /HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/policy/ppo_${TIMESTAMP}.log