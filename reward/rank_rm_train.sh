TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

python /HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/reward/rank_rm_train.py 2>&1 | tee /HOME/sustc_yqzhang/sustc_yqzhang_1/sy/RLHF/reward/rank_rm_${TIMESTAMP}.log