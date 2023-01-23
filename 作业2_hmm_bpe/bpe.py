# 任务二:BPE算法用于英文分词

"""
任务二评分标准:

1. 共有7处TODO需要填写,每个TODO计1-2分,共9分,预计代码量50行；
2. 允许自行修改、编写代码完成,对于该情况,请补充注释以便于评分,否则结果不正确将导致较多的扣分；
3. 实验报告(python)/用于说明实验的文字块(jupyter notebook)不额外计分,但不写会导致扣分。

"""


import re
import functools


# 构建空格分词器,将语料中的句子以空格切分成单词,然后将单词拆分成字母加`</w>`的形式。例如`apple`将变为`a p p l e </w>`。
_splitor_pattern = re.compile(r"[^a-zA-Z']+|(?=')")
_digit_pattern = re.compile(r"\d+")

def white_space_tokenize(corpus):
    """
    先正则化(字母转小写、数字转为N、除去标点符号),然后以空格分词语料中的句子,例如:
    输入 corpus=["I am happy.", "I have 10 apples!"],
    得到 [["i", "am", "happy"], ["i", "have", "N", "apples"]]

    Args:
        corpus: List[str] - 待处理的语料

    Return:
        List[List[str]] - 二维List,内部的List由每个句子的单词str构成
    """

    tokeneds = [list(
        filter(lambda tkn: len(tkn)>0, _splitor_pattern.split(_digit_pattern.sub("N", stc.lower())))) for stc in corpus
    ]
    
    return tokeneds


# 编写相应函数构建BPE算法需要用到的初始状态词典
def build_bpe_vocab(corpus):
    """
    将语料进行white_space_tokenize处理后,将单词每个字母以空格隔开、结尾加上</w>后,构建带频数的字典,例如:
    输入 corpus=["I am happy.", "I have 10 apples!"],
    得到
    {
        'i </w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
     }

    Args:
        corpus: List[str] - 待处理的语料

    Return:
        Dict[str, int] - "单词分词状态->频数"的词典
    """

    tokenized_corpus = white_space_tokenize(corpus)

    bpe_vocab = dict()
    
    # TODO: 完成函数体(1分)
    for sentence in tokenized_corpus:
        for word in sentence:
            key = str()
            for letter in word:
                key += letter
                key += ' '
            key += '</w>'
            if key not in bpe_vocab.keys():
                bpe_vocab[key] = 1
            else:
                bpe_vocab[key] += 1 
    return bpe_vocab


# 编写所需的其他函数
def get_bigram_freq(bpe_vocab):
    """
    统计"单词分词状态->频数"的词典中,各bigram的频次(假设该词典中,各个unigram以空格间隔),例如:
    输入 bpe_vocab=
    {
        'i </w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
    }
    得到
    {
        ('i', '</w>'): 2,
        ('a', 'm'): 1,
        ('m', '</w>'): 1,
        ('h', 'a'): 2,
        ('a', 'p'): 2,
        ('p', 'p'): 2,
        ('p', 'y'): 1,
        ('y', '</w>'): 1,
        ('a', 'v'): 1,
        ('v', 'e'): 1,
        ('e', '</w>'): 1,
        ('N', '</w>'): 1,
        ('p', 'l'): 1,
        ('l', 'e'): 1,
        ('e', 's'): 1,
        ('s', '</w>'): 1
    }

    Args:
        bpe_vocab: Dict[str, int] - "单词分词状态->频数"的词典

    Return:
        Dict[Tuple(str, str), int] - "bigram->频数"的词典
    """

    bigram_freq = dict()
    
    # TODO: 完成函数体(1分)
    for word, freq in bpe_vocab.items():
        word_split = word.split(' ')
        for i in range(len(word_split) - 1):
            data = tuple(word_split[i:i + 2])
            if data in bigram_freq:
                bigram_freq[data] += freq
            else:
                bigram_freq[data] = freq

    return bigram_freq


def refresh_bpe_vocab_by_merging_bigram(bigram, old_bpe_vocab):
    """
    在"单词分词状态->频数"的词典中,合并指定的bigram(即去掉对应的相邻unigram之间的空格),最后返回新的词典,例如:
    输入 bigram=('i', '</w>'),old_bpe_vocab=
    {
        'i </w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
    }
    得到
    {
        'i</w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
    }

    Args:
        old_bpe_vocab: Dict[str, int] - 初始"单词分词状态->频数"的词典

    Return:
        Dict[str, int] - 合并后的"单词分词状态->频数"的词典
    """
    
    new_bpe_vocab = dict()

    # TODO: 完成函数体(1分)
    for word, freq in old_bpe_vocab.items():
        i = 0
        word_replace = []
        word_split = word.split(' ')
        while i < len(word_split):
            if i != len(word_split) - 1:
                if bigram ==tuple(word_split[i:i+2]):
                    word_replace.append(bigram[0] + bigram[1])
                    i = i + 1
                else:
                    word_replace.append(word_split[i])
            else:
                word_replace.append(word_split[i])
            i = i + 1
        new_data = ''
        for i in range(len(word_replace)-1):
            new_data += word_replace[i]
            new_data += ' '
        new_data += (word_replace[-1])
        new_bpe_vocab[new_data] = freq
    
    return new_bpe_vocab


def get_bpe_tokens(bpe_vocab):
    """
    根据"单词分词状态->频数"的词典,返回所得到的BPE分词列表,并将该列表按照分词长度降序排序返回,例如:
    输入 bpe_vocab=
    {
        'i</w>': 2,
        'a m </w>': 1,
        'ha pp y </w>': 1,
        'ha v e </w>': 1,
        'N </w>': 1,
        'a pp l e s </w>': 1
    }
    得到
    [
        ('i</w>', 2),
        ('ha', 2),
        ('pp', 2),
        ('a', 2),
        ('m', 1),
        ('</w>', 5),
        ('y', 1),
        ('v', 1),
        ('e', 2),
        ('N', 1),
        ('l', 1),
        ('s', 1)
     ]

    Args:
        bpe_vocab: Dict[str, int] - "单词分词状态->频数"的词典

    Return:
        List[Tuple(str, int)] - BPE分词和对应频数组成的List
    """
    
    # TODO: 完成函数体(2分)
    bpe_tokens = []
    tokens_dict = dict()
    for word,freq in bpe_vocab.items():
        word_split = word.split(' ')
        for gram in word_split:
            if gram in tokens_dict.keys():
                tokens_dict[gram] += freq
            else:
                tokens_dict[gram] = freq
    for word,freq in tokens_dict.items():
        bpe_tokens.append((word,freq))
    bpe_tokens = sorted(bpe_tokens, key=lambda x: len(x[0]), reverse=True)
    return bpe_tokens


def print_bpe_tokenize(word, bpe_tokens):
    """
    根据按长度降序的BPE分词列表,将所给单词进行BPE分词,最后打印结果。
    
    思想是,对于一个待BPE分词的单词,按照长度顺序从列表中寻找BPE分词进行子串匹配,
    若成功匹配,则对该子串左右的剩余部分递归地进行下一轮匹配,直到剩余部分长度为0,
    或者剩余部分无法匹配(该部分整体由"<unknown>"代替)
    
    例1:
    输入 word="supermarket", bpe_tokens=[
        ("su", 20),
        ("are", 10),
        ("per", 30),
    ]
    最终打印 "su per <unknown>"

    例2:
    输入 word="shanghai", bpe_tokens=[
        ("hai", 1),
        ("sh", 1),
        ("an", 1),
        ("</w>", 1),
        ("g", 1)
    ]
    最终打印 "sh an g hai </w>"

    Args:
        word: str - 待分词的单词str
        bpe_tokens: List[Tuple(str, int)] - BPE分词和对应频数组成的List
    """
    # TODO: 请尝试使用递归函数定义该分词过程(2分)
    def bpe_tokenize(sub_word):
        if sub_word == '':
            return sub_word
        
        for word, freq in bpe_tokens:
            length = len(sub_word) - len(word)
            for i in range(0, length + 1):
                if sub_word[i: i+len(word)] == word:
                    return bpe_tokenize(sub_word[: i]) + word + ' ' + bpe_tokenize(sub_word[i+len(word):])
        return '<unknown>'

    res = bpe_tokenize(word+"</w>")
    print(res)


# 开始读取数据集并训练BPE分词器
with open("data/news.2007.en.shuffled.deduped.train", encoding="utf-8") as f:
    training_corpus = list(map(lambda l: l.strip(), f.readlines()[:1000]))

print("Loaded training corpus.")

training_iter_num = 300

training_bpe_vocab = build_bpe_vocab(training_corpus)
for i in range(training_iter_num):
    # TODO: 完成训练循环内的代码逻辑(2分)
    bigram_freq = get_bigram_freq(training_bpe_vocab)
    best_freq_bigram = ''
    best_freq = 0
    for bigram, freq in bigram_freq.items():
        if freq > best_freq:
            best_freq = freq
            best_freq_bigram = bigram
    training_bpe_vocab = refresh_bpe_vocab_by_merging_bigram(best_freq_bigram, training_bpe_vocab)

training_bpe_tokens = get_bpe_tokens(training_bpe_vocab)


# 测试BPE分词器的分词效果
test_word = "naturallanguageprocessing"

print("naturallanguageprocessing 的分词结果为:")
print_bpe_tokenize(test_word, training_bpe_tokens)
