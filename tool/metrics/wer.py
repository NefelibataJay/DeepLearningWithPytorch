import numpy as np

import jieba


class WordErrorRate:
    def __init__(self, lang):
        self.lang = lang

    def cut_sentence(self, sentence):
        if self.lang == 'zh':
            return list(jieba.cut(sentence))
        elif self.lang == 'en':
            return sentence.split(' ')
        else:
            raise Exception('Unsupported language.')

    def __call__(self, reference, hypothesis):
        """
        计算语音识别的词错率

        Args:
            reference: str，正确文本
            hypothesis: str，语音识别结果

        Returns:
            float，词错率
        """
        # 分词
        ref_words = self.cut_sentence(reference)
        hyp_words = self.cut_sentence(hypothesis)

        # 初始化变量
        r = ref_words
        h = hyp_words
        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # 动态规划计算编辑距离
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        # 计算词错率
        wer = float(d[len(r)][len(h)]) / float(len(r))

        return wer
