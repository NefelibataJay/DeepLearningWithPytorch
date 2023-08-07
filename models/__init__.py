from models.conformer_ctc import ConformerCTC
from models.conformer_ctc_attention import ConformerCTCAttention
from models.conformer_transducer import ConformerTransducer
from models.branchformer_ctc import BranchformerCTC

REGISTER_MODEL = {
    "conformer_ctc": ConformerCTC,
    "conformer_transducer": ConformerTransducer,
    "conformer_ctc_attention": ConformerCTCAttention,
    "branchformer_ctc": BranchformerCTC,
}
