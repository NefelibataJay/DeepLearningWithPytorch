import torch

from tool.optimizer import WarmupLR
from tool.optimizer.gradual_warmup_lr import GradualWarmupScheduler
from matplotlib import pyplot as plt

if __name__ == "__main__":
    max_epoch = 100
    warmup_epoch = 20
    lr = 0.001
    accum_grad = 4
    data_num = 100
    grad_clip =5
    eval_steps = 20
    T_max = max_epoch - warmup_epoch

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epoch, after_scheduler=scheduler)
    lrs = []
    losses = []
    for epoch in range(max_epoch):
        print("Epoch:", epoch)
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for idx, batch in enumerate(range(1, data_num)):
            inputs = torch.randn(10, 10)
            targets = torch.randn(10, 10)
            inputs = inputs.to("cpu")
            targets = targets.to("cpu")
            logits = model(inputs)

            loss = torch.nn.functional.mse_loss(logits, targets)

            loss /= accum_grad
            loss.backward()

            if (idx + 1) % accum_grad == 0 or (idx + 1) == data_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= data_num
        print("train_loss:", train_loss, "train_lr", scheduler.get_last_lr())
        lrs.append(scheduler.get_last_lr())
        losses.append(train_loss)

        scheduler.step()
        if (epoch + 1) % eval_steps == 0:
            print("valid")
    iters = [i for i in range(max_epoch)]

    plt.plot(iters, lrs)
    # plt.plot(iters, losses)
    plt.show()
