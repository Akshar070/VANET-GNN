import torch
from src.models.gae import GAEEncoder, VANETGraphAutoencoder

def test_gae_encoder_shape():
    encoder = GAEEncoder(3, 64, 16)
    x = torch.randn((10, 3))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    z = encoder(x, edge_index)
    assert z.shape == (10, 16)

def test_gae_decoder_shape():
    encoder = GAEEncoder(3, 64, 16)
    model = VANETGraphAutoencoder(encoder)
    z = torch.randn((10, 16))
    adj_hat = model.decode_all(z)
    assert adj_hat.shape == (10, 10)

def test_gae_recon_loss_finite():
    encoder = GAEEncoder(3, 64, 16)
    model = VANETGraphAutoencoder(encoder)
    z = torch.randn((10, 16))
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    loss = model.recon_loss(z, edge_index)
    assert torch.isfinite(loss)

def test_gae_no_edges():
    encoder = GAEEncoder(3, 64, 16)
    model = VANETGraphAutoencoder(encoder)
    z = torch.randn((10, 16))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    loss = model.recon_loss(z, edge_index)
    assert torch.isfinite(loss)
