from torchmetrics.text import CharErrorRate


def test_cer():
    cer = CharErrorRate(ignore_case=True, reduction="mean")
    preds = ["this ", "there is an other"]
    target = ["this ", "there is another"]
    cer(preds, target)
    print(cer)
    print(cer.compute() * 100)


if __name__ == "__main__":
    test_cer()
