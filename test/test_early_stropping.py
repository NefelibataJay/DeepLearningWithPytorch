from trainer.early_stopping import EarlyStopping


def test_early_stropping():
    early_stopping = EarlyStopping(save_path="../outputs/checkpoints")

    for i in range(50):
        early_stopping(i, None)
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    test_early_stropping()
