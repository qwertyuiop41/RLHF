TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

python reward/rank_rm_train.py 2>&1 | tee reward/rank_rm_${TIMESTAMP}.log