"""Pools of user utterances used to construct fake-interruption rounds.

Two kinds of fake-interruption user utterances are supported:

- **Affirmations** (Chinese): short agreement / acknowledgment phrases. They
  are split into three buckets by frequency so the data distribution biases
  toward the most natural short replies.
- **Unrelated speech**: "talking to others" / asides the user might say while
  the assistant is mid-utterance, which the assistant should ignore.

These pools are intentionally short and easy to extend; add your own entries
to bias toward your target domain or speaker style.
"""

import random
from typing import List


HIGH_FREQUENCY_AFFIRMATIONS: List[str] = [
    "对", "没错", "是的", "确实", "很对", "正解", "有道理", "赞同", "你说的很有道理呢",
    "确实是这样的", "我非常认同你的观点", "你说的没错，我也这么觉得", "对呀，和我想的一样",
    "确实如此，很有见地", "挺对的，我也这么看", "很有道理，我完全同意", "你说得太对啦",
    "完全正确", "非常正确", "我也这么认为", "你的想法很正确", "确实很对", "对极了",
    "你说的在理", "很在理", "我和你意见一致", "你说的挺靠谱", "这话说得对", "很符合我的想法",
    "和我的看法相同", "你的看法很中肯", "非常赞同你的说法", "你说的很符合实际",
    "你分析得很对", "确实很有道理",
]

MEDIUM_FREQUENCY_AFFIRMATIONS: List[str] = [
    "我认可你的说法", "对，很有道理", "没错，很合理", "是这样的，很对", "确实对，很认同",
    "有道理，很正确", "你的话很有说服力", "很正确的观点", "你说的确实没错", "非常对，我同意",
    "我觉得你说得很对", "你讲得很有道理", "确实是这么回事", "你说的很对路",
    "完全符合我的观点", "你的观点很准确", "很恰当的观点", "我完全站在你这边",
    "你的想法无可挑剔", "很有见地的话", "你说的很符合常理", "非常在理的说法",
    "你的见解很独到", "确实很符合逻辑", "我同意你的想法", "你说的话很中听",
    "很有道理的见解", "你的看法很有深度", "确实是个好观点", "你说得很有分寸",
    "很正确的思路", "你的话语很合理", "我认可你的意见", "很符合事实",
]

LOW_FREQUENCY_AFFIRMATIONS: List[str] = [
    "你的观点很正", "很有说服力的观点", "你说的很到位", "确实是个正确的看法",
    "我很赞同你的思路", "你的想法很清晰", "很有条理的观点", "你说的很有依据",
    "确实是合理的想法", "你的话很正确", "我赞成你的看法", "你表达得很精准",
    "很符合道理", "你的观点很有价值", "确实是好想法", "你说的很客观",
    "很符合实际情况", "你的见解很正确", "确实是真理", "你的想法很完美",
    "很有意义的观点", "你说的很对味", "很符合我的心意", "你的观点很可靠",
]

TALKING_TO_OTHERS: List[str] = [
    "这杯咖啡太甜了。", "猫躺在沙发上。", "风吹得有点凉。", "今天的阳光真好。",
    "花已经开了。", "书桌上有一杯水。", "鞋子有点挤脚。", "刚刚听到鸟叫了。",
    "窗外的草地真绿。", "桌子上放着一个苹果。", "灯光有点刺眼。", "地板刚拖得很干净。",
    "水壶的水满了。", "鱼在水里游来游去。", "窗帘被风吹起来了。", "路边的树都变黄了。",
    "刚才掉了一本书。", "门被风吹开了。", "汤还是太烫了。", "椅子有点不稳。",
    "草地上有好多落叶。", "这个枕头太软了。", "花盆里的土干了。", "房间里的空气很好。",
    "衣服被洗得很干净。", "刚刚点燃了一根蜡烛。", "手机壳有点脏了。", "这个橘子特别甜。",
    "窗外的小鸟飞走了。", "风铃发出了清脆的声音。", "天上的云很漂亮。", "地上的水干了。",
    "沙发靠垫有点旧了。", "书架上的书倒了。", "车开得很快。", "昨晚的月亮特别圆。",
    "门口的鞋有点乱。", "新买的拖鞋特别舒服。", "草丛里有一只小猫。", "家里的小狗在打盹。",
    "咖啡有点凉了。", "听到了楼下的音乐。", "墙上的挂画很好看。", "碗里的汤还没喝完。",
    "地毯的颜色真鲜艳。", "枕头很蓬松。", "窗帘挡住了阳光。", "电脑桌太乱了。",
    "手表时间不准了。", "杯子裂了一道口子。", "草坪被修剪得很整齐。", "阳台上的衣服晾干了。",
    "外面的风渐渐停了。", "书包拉链坏了。", "最近的夜晚特别安静。", "天空的颜色渐渐变深。",
    "新买的耳机很舒服。", "鞋底沾了好多泥。", "楼道里很安静。", "花瓶里的花有点蔫了。",
    "外套的扣子掉了。", "墙角有一点灰。", "窗外的景色真不错。", "杯子里的水少了。",
    "刚刚收到了快递。", "新的地毯很柔软。", "鱼缸的水需要换了。", "路边的小花开了。",
    "早餐吃得很满足。", "小朋友在玩泥巴。", "客厅里的灯关了。", "家里的猫咪在发呆。",
    "衣服上的纽扣掉了。", "车轮压过了水坑。", "鞋底的泥干了。", "草坪边的长椅很干净。",
    "雨后的空气很清新。", "电脑的风扇很安静。", "桌上的笔记本被打开了。", "墙上的挂钟停了。",
    "屋里的蜡烛点亮了。", "外面的树叶掉光了。", "家里的地板很滑。", "阳光从窗帘缝里透了进来。",
    "鱼缸里的鱼特别活跃。", "这个杯子放得很稳。", "街道上的人渐渐少了。", "手上的水还没擦干。",
    "厨房的垃圾桶需要清理了。", "阳台上的植物长高了。", "客厅里的沙发很整齐。",
]


_BUCKETS = [HIGH_FREQUENCY_AFFIRMATIONS, MEDIUM_FREQUENCY_AFFIRMATIONS, LOW_FREQUENCY_AFFIRMATIONS]
_BUCKET_WEIGHTS = [0.5, 0.35, 0.15]


def sample_affirmation(rng: random.Random | None = None) -> str:
    """Sample an affirmation, weighted by frequency bucket."""
    rng = rng or random
    bucket = rng.choices(_BUCKETS, weights=_BUCKET_WEIGHTS, k=1)[0]
    return rng.choice(bucket)


def sample_unrelated_speech(rng: random.Random | None = None) -> str:
    """Sample an unrelated barge-in utterance."""
    rng = rng or random
    return rng.choice(TALKING_TO_OTHERS)


def load_fake_query_pool(path: str) -> List[str]:
    """Load a custom pool of fake-interruption queries from a text file.

    Each line is treated as one entry. Empty lines are ignored. Use this to
    supply a domain-specific pool (e.g. distilled from real interaction logs).
    """
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(line)
    return queries
