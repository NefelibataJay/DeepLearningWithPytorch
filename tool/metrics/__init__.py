from torchmetrics.text import CharErrorRate

REGISTERED_METRICS = {
    'cer': CharErrorRate,
}
