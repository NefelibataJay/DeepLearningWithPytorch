from models.conformer_ctc import ConformerCTC
from models.conformer_transducer import ConformerTransducer

REGISTER_MODEL = {
    "conformer_ctc": ConformerCTC,
    "conformer_transducer": ConformerTransducer,
}
