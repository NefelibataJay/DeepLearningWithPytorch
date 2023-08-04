from torchmetrics.text import CharErrorRate


def test_cer():
    cer = CharErrorRate()
    preds = ["什么辣鸡", "为什么1"]
    target = ["什么垃圾", "为什么1"]
    cer(preds, target)
    print(cer.compute() * 100)

    cer = CharErrorRate()
    cer(preds[0], target[0])
    print(cer.compute() * 100)

    cer = CharErrorRate()
    cer(preds[1], target[1])
    print(cer.compute() * 100)

if __name__ == "__main__":
    test_cer()
