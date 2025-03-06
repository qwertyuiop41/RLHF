def correctness_reward(prompts, completions, answer, **kwargs):
    """
    返回准确率的奖励
    """

    responses = [completion[0]['content'] for completion in completions]

    return rewards

def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.
    Also logs detailed format compliance metrics.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    format_scores = []

    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.20
        if "</reasoning>" in response: score += 0.20
        if "<answer>" in response: score += 0.20
        if "</answer>" in response: score += 0.20
        rewards.append(score)
        format_scores.append(score)

    return rewards