

TIMESTAMP2=$(date +"%Y-%m-%d_%H-%M-%S")

python grpo/train2.py 2>&1 | tee grpo/grpo2_${TIMESTAMP2}.log