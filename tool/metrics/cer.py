import numpy as np


class CharErrorRate:
    def __call__(self, reference, hypothesis):
        """
            计算中文语音识别的字错率

            Args:
                reference: str，正确文本
                hypothesis: str，语音识别结果

            Returns:
                float，字错率
            """
        # 初始化变量
        r = reference.replace(' ', '')
        h = hypothesis.replace(' ', '')
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
        return float(d[len(r)][len(h)]) / float(len(r))
