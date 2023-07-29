from models.decoder.transducer import TransducerJoint, RnnPredictor
import torch


def test_transducer():
    encoder_dim = 16
    predictor_dim = 16
    joint_dim = 10
    vocab_size = 15
    batch_size = 1
    blank = 3
    sos = 1
    eos = 2
    predictor = RnnPredictor(vocab_size=vocab_size, embed_size=10, hidden_size=20, output_size=predictor_dim,
                             num_layers=2)
    joint = TransducerJoint(vocab_size, encoder_dim, predictor_dim, joint_dim)

    output_length = 11
    encoder_output = torch.randn(batch_size, output_length, encoder_dim)
    pred_input_step = encoder_output.new_zeros(batch_size, 1).fill_(sos).long()  # [[sos_id]]
    # first decode input is sos
    t = 0
    hyps = list()
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = 64
    per_frame_noblk = 0
    while t < output_length:
        encoder_out_step = encoder_output[:, t:t + 1, :]
        if prev_out_nblk:
            pred_out_step, hidden_states = predictor(pred_input_step)
            # decoder_outputs (batch_size, 1, decoder_output_size)

        joint_out_step = joint(encoder_out_step, pred_out_step)
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()

        if joint_out_max != blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
        if joint_out_max == blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == blank:
                prev_out_nblk = False
            t = t + 1
            per_frame_noblk = 0

    print(hyps)


if __name__ == "__main__":
    test_transducer()
