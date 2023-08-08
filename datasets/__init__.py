from datasets.aishell import AishellDataset
from datasets.asr_dataset import ASRDataset

REGISTER_DATASET = {
    "aishell": AishellDataset,
    "asr_dataset": ASRDataset,
}
