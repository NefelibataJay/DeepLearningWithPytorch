from torchmetrics import CharErrorRate

REGISTERED_METRICS = {
    'cer': CharErrorRate,
}
