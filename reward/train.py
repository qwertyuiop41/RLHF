from rm import RewardModel,RankRewardModel


from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Union, Iterable
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import gc
from datasets import dataset_dict, load_from_disk
# from datasets import Dataset
from tqdm import trange
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM,
    GPT2Tokenizer,
    GPT2Model,
    Qwen2Config
)
from torch.utils.data import DataLoader,Dataset


def rank_data_prepare(pretrain_path):
    data_outputs = {
        'input_ids': [],
        'attention_mask': []
    }
    #越前面分越高
    data_lst = [
        [
            '我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是现存最大、保存最完整的动物，属于国家二级保护动物。熊猫种类繁多，分布广泛，主要分布在四川、云南、陕西、甘肃、宁夏、内蒙古、新疆、青海、吉林、辽宁、黑龙江、山西、江苏、江西、河南、湖北、湖南、广东、广西、海南、重庆、贵州、西藏、四川等省区市。它们的栖息地主要为亚热带或热带的（低地）湿润低地林、亚高山草原、高山湖泊、高原湿润山区和高原沼泽地等，常栖息在高海拔地区。在中国大陆，熊猫分布于四川省甘孜藏族自治州和青海省西宁市等地。雄性熊猫体长约1.5米', 
            '我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是今世界上保存最完好的哺乳动物之一，也是世界自然保护联盟濒危物种红色名录的保护对象之一。在这里，你可以看到全世界最大的熊猫栖息地成都。成都是中国国家林业局直属的国家重点风景名胜区，是国家森林公园、国家湿地公园和国家地质公园的重要组成部分，是全国重点文物保护单位、全国生态文明建设示范区、中国红色旅游名城、国际生态旅游目的地和国际旅游岛建设先进区。地址：四川省成都市绵阳市成华区成都高新技术产业开发区成华大道1号乘车路线：成都绵阳都江堰雅', 
            '我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家二级保护动物，是世界上保存最完整的动物种群之一，也是我国第一个国家级自然保护区。我们是四川省的首批国家重点保护野生动物和珍稀动物基金会的成员，被誉为中国动物保护的摇篮和世界生物多样性保护基地，被中国科学院、中华人民共和国国家林业局授予全国生态文明建设示范区称号，被国务院批准为国家森林城市、国际生态旅游目的地。熊猫基地位于成都市双流区东南部，是国家aaaa级旅游景区，国家地理标志保护单位。熊猫栖息地为亚热带或热带的高山', 
            '我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家级自然保护区，也是世界上保存最完整、规模最大的野生动物种类繁多的地区之一，是中国国家重点保护的珍稀濒危动物及其栖息地和世界自然遗产的重要组成部分，被誉为中国最美丽的城市和世界生物多样性保护基地，被国际旅游组织评为全球生态旅游目的地。成都熊猫国家公园位于四川省甘孜藏族自治州，是国家aaaa级旅游景区，被《世界遗产名录》列为全国重点文物保护单位。目前，我国已建成国家森林公园、国家湿地公园和国家地质公园，国家林业局、国务院扶贫',
            '我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家级自然保护区，也是世界上保存最完好的熊猫种群之一。它们栖息在亚热带或热带的高海拔草原上，生活环境十分优越，是中国四大自然奇观之一，被誉为世界自然遗产和中国国家森林公园。熊猫栖息地主要分布在中国大陆的西藏、青海、甘肃、宁夏、新疆、内蒙古、山西、辽宁、吉林、黑龙江、江苏、河南、安徽、湖北、湖南、江西、广东、海南、四川、云南、贵州、陕西等地。中国熊猫研究中心主任、中国科学院院士、国家自然科学基金委员会委员、中华全国工商业联合会副主席'
        ], 
        [
            '昨天买的，今天就到了，因为给家中父母买的，怕东西多老人取件不方便，今天听家里人说京东小哥送到家门楼下，心里太高兴了，在这里希望京东能表扬一下本次快递小哥，他让我本次购物感觉很好，本来就喜欢京东一直购物，现在我更欣赏。购物的同事还能享受温暖的服务，京东的快递服务果然很棒，在此感谢京东，感觉快递小哥，如此服务真的很温暖。', 
            '京东 ，对于S8的货品状态 ，你们你们京东采购下单是应该在预售前还是预售后(定金不退的预售方式)？预售前下单叫正规预订补款了有货拿，预售补款了没货并且还要重新再采购叫空手套白狼，京东是哪种？', 
            '在北京住过不下10多家酒店，也喜欢住公寓，从凯宾斯基到建国饭店，从京广到美华再到星城亮马，而这个是我住过的有史以来最差的一个酒店公寓。难怪价格不上不下，不是因为临时有事绝对不住，希望这里那么多好评语不是枪手1、入口难找到要死不说，大堂感觉就是某个买小商品的商铺，check in 竟然要压证件，没有听说过，坚决不同意拿了我的证件去复印。私人住宿和旅客混杂，拖着箱子看着买菜回来的人一同电梯很奇怪。2、半夜接到骚扰电话3、房间设计装饰非常的“家常“，设施陈旧，非常像当年在江南古镇租住的农家房3、住的房间刚好在过道口，声音那叫一个大阿，谁说的房间隔音？楼上住户的动静镇清楚啊4、服务态度不好，和客人顶着说，铁板一样的语气。5， 实在要找一优点出来的话：唯一就是小区里面比较安静，没有汽车闹声。', 
            '码数刚刚好，穿上很好看，和身。宝贝不掉色，弹力好。穿着不紧绷，试了好几下蹲下站起来，都轻松自如，不会感觉腿被束缚着。价格也不贵，现在认准这家店了这款洗发水挺适合我的发质，用完果断续上一瓶，还搞了个特价，值了！', 
            '之前就听说苏州万丽是苏州生意最好，房价最高，也是业内人士最推崇的酒店，远胜于喜来登，香格里拉，索菲特，在苏州属于一枝独秀型的，平时房间非常的难定，几乎天天满房，这次好不容易定了个行政套，本打算住一天，后又延了一天，简单来说吧，房间不大但很温馨，酒店工作人员不多但都非常专业，亲切，严格意义上来说该酒店硬件并不突出，没有游泳池，没有特色餐厅，建筑也没有什么特色，处处透露着简单，适用，大气，但是只有你住了以后才会觉得，值！'
        ]
    ]

    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

    #越前面分越高
    for batch_text in data_lst:
        text_encode=tokenizer(batch_text, max_length=256, padding="max_length", truncation=True,return_tensors='pt')
        data_outputs["input_ids"].append(text_encode["input_ids"])
        data_outputs["attention_mask"].append(text_encode["attention_mask"])

    print(data_outputs)
    return data_outputs,tokenizer






def data_prepare(pretrain_path):
    data_lst = [
        "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是今世界上保存最完好的哺乳动物之一，也是世界自然保护联盟濒危物种红色名录的保护对象之一。在这里，你可以看到全世界最大的熊猫栖息地成都。成都是中国国家林业局直属的国家重点风景名胜区，是国家森林公园、国家湿地公园和国家地质公园的重要组成部分，是全国重点文物保护单位、全国生态文明建设示范区、中国红色旅游名城、国际生态旅游目的地和国际旅游岛建设先进区。地址：四川省成都市绵阳市成华区成都高新技术产业开发区成华大道1号乘车路线：成都绵阳都江堰雅",
        "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家二级保护动物，是世界上保存最完整的动物种群之一，也是我国第一个国家级自然保护区。我们是四川省的首批国家重点保护野生动物和珍稀动物基金会的成员，被誉为中国动物保护的摇篮和世界生物多样性保护基地，被中国科学院、中华人民共和国国家林业局授予全国生态文明建设示范区称号，被国务院批准为国家森林城市、国际生态旅游目的地。熊猫基地位于成都市双流区东南部，是国家aaaa级旅游景区，国家地理标志保护单位。熊猫栖息地为亚热带或热带的高山",
        "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家级自然保护区，也是世界上保存最完好的熊猫种群之一。它们栖息在亚热带或热带的高海拔草原上，生活环境十分优越，是中国四大自然奇观之一，被誉为世界自然遗产和中国国家森林公园。熊猫栖息地主要分布在中国大陆的西藏、青海、甘肃、宁夏、新疆、内蒙古、山西、辽宁、吉林、黑龙江、江苏、河南、安徽、湖北、湖南、江西、广东、海南、四川、云南、贵州、陕西等地。中国熊猫研究中心主任、中国科学院院士、国家自然科学基金委员会委员、中华全国工商业联合会副主席",
        "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家级自然保护区，也是世界上保存最完整、规模最大的野生动物种类繁多的地区之一，是中国国家重点保护的珍稀濒危动物及其栖息地和世界自然遗产的重要组成部分，被誉为中国最美丽的城市和世界生物多样性保护基地，被国际旅游组织评为全球生态旅游目的地。成都熊猫国家公园位于四川省甘孜藏族自治州，是国家aaaa级旅游景区，被《世界遗产名录》列为全国重点文物保护单位。目前，我国已建成国家森林公园、国家湿地公园和国家地质公园，国家林业局、国务院扶贫",
        "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是现存最大、保存最完整的动物，属于国家二级保护动物。熊猫种类繁多，分布广泛，主要分布在四川、云南、陕西、甘肃、宁夏、内蒙古、新疆、青海、吉林、辽宁、黑龙江、山西、江苏、江西、河南、湖北、湖南、广东、广西、海南、重庆、贵州、西藏、四川等省区市。它们的栖息地主要为亚热带或热带的（低地）湿润低地林、亚高山草原、高山湖泊、高原湿润山区和高原沼泽地等，常栖息在高海拔地区。在中国大陆，熊猫分布于四川省甘孜藏族自治州和青海省西宁市等地。雄性熊猫体长约1.5米"]
    # 自定义打分标签，每个句子一个分值。也可以定义多维度的打分方法，只是模型的线性层需要改为你所定义的维度数
    direct_score = [[0.75], [0.5], [0.35], [0.4], [0.8]]
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    train_data = tokenizer.batch_encode_plus(data_lst, max_length=256, padding="max_length", truncation=True,return_tensors='pt')
    train_data["labels"] = torch.tensor(direct_score)
    return train_data, tokenizer
class Datasets(Dataset):               
    def __init__(self, sample):
        super(Datasets, self).__init__()
        self.sample = sample

    def __getitem__(self, item):
        res = {k: v[item] for k, v in self.sample.items()}
        return res

    def __len__(self):
        return len(self.sample['input_ids'])



def rank_loss(predict_rewards):
    print("====================================")
    # predict_rewards的位置越前面的相对分越高
    # loss设置原因见：https://zhuanlan.zhihu.com/p/610147705
    loss, counts = torch.tensor([0]), 0
    # predict_rewards的位置越前面的相对分越高
    for i in range(len(predict_rewards) - 1):  # 遍历所有前项-后项的得分差
        for j in range(i + 1, len(predict_rewards)):
            diff = nn.functional.logsigmoid(predict_rewards[i] - predict_rewards[j])  # sigmoid到0~1之间 log再全变成负的
            print(f"diff:{diff}")
            loss = loss + diff
            counts += 1
            print(loss)
    loss =loss / counts
    return -loss  # 要最大化分差，所以要取负数

def train(pretrain_path, save_path):
    config = Qwen2Config.from_pretrained(pretrain_path)
    model = RewardModel(config=config)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    train_data, tokenizer = data_prepare(pretrain_path)
    print(type(train_data))
    print(train_data)
    exit()
    dataloader = DataLoader(dataset=Datasets(train_data), shuffle=False, batch_size=1)

    max_train_steps = 10 * len(dataloader)
    warm_steps = int(0.0 * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    model.train()
    for i in range(1, 51):
        loss_lst = []
        for batch in dataloader:
            out, loss = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss_lst.append(loss.item())
            print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print("epoch{}\tloss: {}".format(str(i), str(sum(loss_lst) / len(loss_lst))))
    tokenizer.save_pretrained(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    model_to_save.config.save_pretrained(save_path)

def rank_train(pretrain_path, save_path):
    config = Qwen2Config.from_pretrained(pretrain_path)
    model = RankRewardModel(config=config)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    train_data, tokenizer = rank_data_prepare(pretrain_path)
    dataloader = DataLoader(dataset=Datasets(train_data), shuffle=False, batch_size=1)

    max_train_steps = 10 * len(dataloader)
    warm_steps = int(0.0 * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    model.train()
    for i in range(1, 2):
        loss_lst = []
        
        for batch in dataloader:
            reward_lst=[]
            print(batch["input_ids"][0])
            print(len(batch["input_ids"][0]))
            for i in range(len(batch["input_ids"][0])):
                reward = model(batch["input_ids"][0][i].unsqueeze(0), attention_mask=batch["attention_mask"][0][i].unsqueeze(0))
                
                print(reward[0])
                reward_lst.append(reward)
            print(reward_lst)
            loss=rank_loss(reward_lst)
            print(loss)
            loss_lst.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print("epoch{}\tloss: {}".format(str(i), str(sum(loss_lst) / len(loss_lst))))
    tokenizer.save_pretrained(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    model_to_save.config.save_pretrained(save_path)


def predict(model_path):
    text = ["我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是今世界上保存最完好的哺乳动物之一，也是世界自然保护联盟濒危物种红色名录的保护对象之一。在这里，你可以看到全世界最大的熊猫栖息地成都。成都是中国国家林业局直属的国家重点风景名胜区，是国家森林公园、国家湿地公园和国家地质公园的重要组成部分，是全国重点文物保护单位、全国生态文明建设示范区、中国红色旅游名城、国际生态旅游目的地和国际旅游岛建设先进区。地址：四川省成都市绵阳市成华区成都高新技术产业开发区成华大道1号乘车路线：成都绵阳都江堰雅",
            "我们去成都旅游，必须要去的地方是大熊猫繁殖基地。大熊猫是我国唯一的国家二级保护动物，是世界上保存最完整的动物种群之一，也是我国第一个国家级自然保护区。我们是四川省的首批国家重点保护野生动物和珍稀动物基金会的成员，被誉为中国动物保护的摇篮和世界生物多样性保护基地，被中国科学院、中华人民共和国国家林业局授予全国生态文明建设示范区称号，被国务院批准为国家森林城市、国际生态旅游目的地。熊猫基地位于成都市双流区东南部，是国家aaaa级旅游景区，国家地理标志保护单位。熊猫栖息地为亚热带或热带的高山",]
    model = RewardModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    data = tokenizer.batch_encode_plus(text, max_length=256, padding="max_length", truncation=True,
                                           return_tensors='pt')
    score = model(**data)
    return score


if __name__=="__main__":
    train(pretrain_path='/home/wsy/NLP/RL/Qwen2.5-0.5B-Instruct',save_path='/home/wsy/NLP/RL/RLHF/reward/ckpt')
