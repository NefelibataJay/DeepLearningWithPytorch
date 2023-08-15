import numpy as np
import torch


def remove_blank(labels, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(y, blank=0):
    """
    :param y: 模型输出的log_softmax值  [T, C] T为时刻数，C为类别数(字典大小)
    :param blank: blank的下标
    :return: label_blank: 每个时刻t上最大值对应的下标
    :return: label: 移除blank后的结果
    """
    label_blank = y.argmax(-1)
    return label_blank


def batch_greedy_decode(y_s, blank=0):
    """
    :param y_s: 模型输出的log_softmax值  [B, T, C] B为batch_size，T为时刻数，C为类别数(字典大小)
    :param blank: blank的下标
    :return: label_blank: 每个时刻t上最大值对应的下标
    :return: label: 移除blank后的结果
    """
    label_blank = y_s.argmax(-1)
    return label_blank


def beam_decode(log_y, beam_size=10):
    """
    :param log_y: 模型输出的log_softmax值  [T, C] T为时刻数，C为类别数(字典大小)
    :param beam_size: beam_size
    """
    # y是个二维数组，记录了所有时刻的所有项的概率
    T, V = log_y.shape
    # 初始的beam
    beam = [([], 0)]
    # 遍历所有时刻t
    for t in range(T):
        # 每个时刻先初始化一个new_beam
        new_beam = []
        # 遍历beam
        for prefix, score in beam:
            # 对于一个时刻中的每一项(一共V项)
            for i in range(V):
                # 记录添加的新项是这个时刻的第几项，对应的概率(log形式的)加上新的这项log形式的概率(本来是乘的，改成log就是加)
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                # new_beam记录了对于beam中某一项，将这个项分别加上新的时刻中的每一项后的概率
                new_beam.append((new_prefix, new_score))
        # 给new_beam按score排序
        new_beam.sort(key=lambda x: x[1], reverse=True)
        # beam即为new_beam中概率最大的beam_size个路径
        beam = new_beam[:beam_size]

    return beam


if __name__ == '__main__':
    np.random.seed(1111)
    y = np.random.random([20, 6])
    y_test = torch.from_numpy(y).log_softmax(-1)
    # label = greedy_decode(y_test)
    beam_chosen = beam_decode(y_test, beam_size=10)
    for beam_string, beam_score in beam_chosen[:20]:
        print(beam_string, beam_score)
