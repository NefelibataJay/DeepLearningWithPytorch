import torch

from models.encoder.branchformer_encoder import BranchformerEncoder


def test_branchformer():
    batch_size = 5
    inputs_length = torch.tensor([120, 100, 110, 150, 130])
    input_dim = 80
    inputs = torch.randint(low=0,high=50,size=(batch_size, 80, 150)).transpose(1, 2).to(torch.float32)

    branchformer = BranchformerEncoder(input_dim=input_dim, encoder_dim=128, num_layers=2)
    print(branchformer)
    outputs, outputs_lengths = branchformer(inputs, inputs_length)
