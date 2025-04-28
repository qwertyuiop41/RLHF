import re


def correctness_reward( completions, labels):
    """
    返回准确率的奖励
    """
    print("correctness_reward:")
    
    rewards=[]
    for i in range(len(completions)):
        answer=extract_answer(completions[i])
        score=0.0
        if answer==labels[i]:
            score=1.0
        rewards.append(score)
    print(rewards)
    return rewards

def format_reward(completions,labels):
    """
    返回符合格式的奖励
    """
    print("format_reward")    
    rewards = []

    for completion in completions:
        score = 0.0
        if "<think>" in completion: score += 0.20
        if "</think>" in completion: score += 0.20
        if "<answer>" in completion: score += 0.20
        if "</answer>" in completion: score += 0.20
        rewards.append(score)
    print(rewards)
    return rewards


def extract_answer(completion):
    try:
        # 使用正则表达式查找最后一个 <answer></answer> 标签及其内部的数字
        match = re.findall(r"<answer>([\d.]+)</answer>", completion)
        if match:
            # 提取最后一个匹配项
            extracted_string = match[-1]
            float_value = float(extracted_string)
            return float_value
        else:
            return None  # 如果没有找到匹配项，则返回 None

    except ValueError:
        return None  # 如果转换失败（例如，提取的字符串不是有效的数字），则返回 None
    except Exception as e:
        print(f"发生错误：{e}")
        return None # 捕捉其他异常
    


