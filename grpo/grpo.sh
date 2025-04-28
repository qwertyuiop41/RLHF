

TIMESTAMP2=$(date +"%Y-%m-%d_%H-%M-%S")

python grpo/train.py 2>&1 | tee grpo/grpo_${TIMESTAMP2}.log