TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

python dpo/train.py 2>&1 | tee dpo/dpo_${TIMESTAMP}.log