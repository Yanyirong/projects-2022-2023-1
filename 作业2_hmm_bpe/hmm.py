# 任务一：HMM模型用于中文分词

"""
任务一评分标准：
1. 共有8处TODO需要填写，每个TODO计1-2分，共9分，预计代码量30行；
2. 允许自行修改、编写代码完成，对于该情况，请补充注释以便于评分，否则结果不正确将导致较多的扣分；
3. 实验报告(python)/用于说明实验的文字块(jupyter notebook)不额外计分，但不写会导致扣分。

注：本任务仅在短句子上进行效果测试，因此对概率的计算可直接进行连乘。
在实践中，常先对概率取对数，将连乘变为加法来计算，以避免出现数值溢出的情况。
"""

import pickle
import numpy as np


# 导入HMM参数，初始化所需的起始概率矩阵，转移概率矩阵，发射概率矩阵
with open("hmm_parameters.pkl", "rb") as f:
    hmm_parameters = pickle.load(f)

# 非断字（B）为第0行，断字（I）为第1行
# 发射概率矩阵中，词典大小为65536，以汉字的ord作为行key
start_probability = hmm_parameters["start_prob"]  # shape(2,)
trans_matrix = hmm_parameters["trans_mat"]  # shape(2, 2)
emission_matrix = hmm_parameters["emission_mat"]  # shape(2, 65536)


# TODO: 将input_sentence中的xxx替换为你的姓名（1分）
input_sentence = "剡宜镕是一名优秀的学生"


# 实现viterbi算法，并以此进行中文分词
def viterbi(sent_orig, start_prob, trans_mat, emission_mat):
    """
    viterbi算法进行中文分词

    Args:
        sent_orig: str - 输入的句子
        start_prob: numpy.ndarray - 起始概率矩阵
        trans_mat: numpy.ndarray - 转移概率矩阵
        emission_mat: numpy.ndarray - 发射概率矩阵

    Return:
        str - 中文分词的结果
    """
    
    #  将汉字转为数字表示
    sent_ord = [ord(x) for x in sent_orig]
    
    # `dp`用来储存不同位置每种标注（B/I）的最大概率值
    dp = np.zeros((2, len(sent_ord)), dtype=float)
    
    # `path`用来储存最大概率对应的上步B/I选择
    #  例如 path[1][7] == 1 意味着第8个（从1开始计数）字符标注I对应的最大概率，其前一步的隐状态为1（I）
    #  例如 path[0][5] == 1 意味着第6个字符标注B对应的最大概率，其前一步的隐状态为1（I）
    #  例如 path[1][1] == 0 意味着第2个字符标注I对应的最大概率，其前一步的隐状态为0（B）
    path = np.zeros((2, len(sent_ord)), dtype=int)
    
    #  TODO: 第一个位置的最大概率值计算（1分）
    dp[0][0] = start_prob[0] * emission_mat[0][sent_ord[0]]
    dp[1][0] = start_prob[1] * emission_mat[1][sent_ord[0]]

    #  TODO: 其余位置的最大概率值计算（填充dp和path矩阵）（2分）
    for i in range(1, len(sent_ord)):
        # 填写dp矩阵
        dp_max_0 = max(dp[0][i-1]*trans_mat[0][0], dp[1][i-1]*trans_mat[1][0])
        dp[0][i] = dp_max_0 * emission_mat[0][sent_ord[i]]
        dp_max_1 = max(dp[0][i-1]*trans_mat[0][1], dp[1][i-1]*trans_mat[1][1])
        dp[1][i] = dp_max_1 * emission_mat[1][sent_ord[i]]
        # 填写path矩阵
        if dp[0][i-1]*trans_mat[0][0] < dp[1][i-1]*trans_mat[1][0]:
            path[0][i] = 1
        else:
            path[0][i] = 0
        if dp[0][i-1]*trans_mat[0][1] < dp[1][i-1]*trans_mat[1][1]:
            path[1][i] = 1
        else:
            path[1][i] = 0

    
    #  `labels`用来储存每个位置最有可能的隐状态
    labels = [0 for _ in range(len(sent_ord))]
    
    #  TODO：计算labels每个位置上的值（填充labels矩阵）（1分） 
    # 名字不用分开，因此可以直接打label
    labels[0] = 0
    labels[1] = 0
    for i in range(2, len(labels)):
        if dp[0][i] >= dp[1][i] and path[0][i] == labels[i-1]:
            labels[i] = 0 
        else: 
            labels[i] = 1
    
    #  根据lalels生成切分好的字符串
    sent_split = []
    for idx, label in enumerate(labels):
        if label == 1:
            sent_split += [sent_ord[idx], ord("/")]
        else:
            sent_split += [sent_ord[idx]]
    sent_split_str = "".join([chr(x) for x in sent_split])

    return sent_split_str


# 实现前向算法，计算该句子的概率值
def compute_prob_by_forward(sent_orig, start_prob, trans_mat, emission_mat):
    """
    前向算法，计算输入中文句子的概率值

    Args:0
        sent_orig: str - 输入的句子
        start_prob: numpy.ndarray - 起始概率矩阵
        trans_mat: numpy.ndarray - 转移概率矩阵
        emission_mat: numpy.ndarray - 发射概率矩阵

    Return:
        float - 概率值
    """
    
    #  将汉字转为数字表示
    sent_ord = [ord(x) for x in sent_orig]

    # `dp`用来储存不同位置每种隐状态（B/I）下，到该位置为止的句子的概率
    dp = np.zeros((2, len(sent_ord)), dtype=float)

    # TODO: 初始位置概率的计算（1分）
    dp[0][0] = start_prob[0] * emission_mat[0][sent_ord[0]]
    dp[1][0] = start_prob[1] * emission_mat[1][sent_ord[0]]
    
    # TODO: 先计算其余位置的概率（填充dp矩阵），然后return概率值（1分）
    for i in range(1, len(sent_ord)):
        dp_0 = dp[0][i-1]*trans_mat[0][0] + dp[1][i-1]*trans_mat[1][0]
        dp[0][i] = dp_0 * emission_mat[0][sent_ord[i]]
        dp_1 = dp[0][i-1]*trans_mat[0][1] + dp[1][i-1]*trans_mat[1][1]
        dp[1][i] = dp_1 * emission_mat[1][sent_ord[i]]

    return sum([dp[i][len(sent_ord)-1] for i in range(2)])


# 实现后向算法，计算该句子的概率值
def compute_prob_by_backward(sent_orig, start_prob, trans_mat, emission_mat):
    """
    后向算法，计算输入中文句子的概率值

    Args:
        sent_orig: str - 输入的句子
        start_prob: numpy.ndarray - 起始概率矩阵
        trans_mat: numpy.ndarray - 转移概率矩阵
        emission_mat: numpy.ndarray - 发射概率矩阵

    Return:
        float - 概率值
    """
    
    #  将汉字转为数字表示
    sent_ord = [ord(x) for x in sent_orig]

    # `dp`用来储存不同位置每种隐状态（B/I）下，从结尾到该位置为止的句子的概率
    dp = np.zeros((2, len(sent_ord)), dtype=float)

    # TODO: 终末位置概率的初始化（1分）
    dp[0][-1] = 1
    dp[1][-1] = 1
    
    # TODO: 先计算其余位置的概率（填充dp矩阵），然后return概率值（1分）
    for i in reversed(range(0, len(sent_ord)-1)):
        dp[0][i] = trans_mat[0][0]*dp[0][i+1]*emission_mat[0][sent_ord[i+1]] + trans_mat[0][1]*dp[1][i+1]*emission_mat[1][sent_ord[i+1]]
        dp[1][i] = trans_mat[1][0]*dp[0][i+1]*emission_mat[0][sent_ord[i+1]] + trans_mat[1][1]*dp[1][i+1]*emission_mat[1][sent_ord[i+1]]

    return sum([dp[i][0] * start_prob[i] * emission_mat[i][sent_ord[0]] for i in range(2)])


print("viterbi算法分词结果：", viterbi(input_sentence, start_probability, trans_matrix, emission_matrix))
print("前向算法概率：", compute_prob_by_forward(input_sentence, start_probability, trans_matrix, emission_matrix))
print("后向算法概率：", compute_prob_by_backward(input_sentence, start_probability, trans_matrix, emission_matrix))
