import time

from tqdm import tqdm

if __name__ == "__main__":
    # 创建一个迭代器，例如一个列表或一个range对象
    data = [1, 2, 3, 4, 5]

    # 使用tqdm包装迭代器，并设置描述信息
    bar = tqdm(enumerate(data), desc="Processing")
    for index, value in bar:
        # 在这里进行循环的操作
        # index为索引，value为元素的值
        # 可以在这里更新模型、计算损失等

        # 示例操作：打印索引和值，并追加更新信息
        progress_info = f"Index: {index}, Value: {value}"
        bar.set_postfix(info=progress_info)
        time.sleep(2)

