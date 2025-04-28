TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

python policy/train3.py 2>&1 | tee policy/ppo3_${TIMESTAMP}.log