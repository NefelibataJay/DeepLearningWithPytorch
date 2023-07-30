from models.conformer_ctc import ConformerCTC
from models.conformer_transducer import ConformerTransducer
from models.branchformer_ctc import BranchformerCTC

REGISTER_MODEL = {
    "conformer_ctc": ConformerCTC,
    "conformer_transducer": ConformerTransducer,
    "branchformer_ctc": BranchformerCTC,
}
