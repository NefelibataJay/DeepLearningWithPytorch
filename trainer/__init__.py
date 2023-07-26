from trainer.conformer_ctc_trainer import ConformerCTCTrainer
from trainer.conformer_transducer_trainer import ConformerTransducerTrainer

REGISTER_TRAINER = {
    "conformer_ctc": ConformerCTCTrainer,
    "conformer_transducer": ConformerTransducerTrainer,
}
