import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

class GAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        z = self.conv2(x, edge_index)
        return z

class VANETGraphAutoencoder(torch.nn.Module):
    def __init__(self, encoder):
        super(VANETGraphAutoencoder, self).__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
        
    def decode(self, z, edge_index):
        # Edge-based decoder: score(i, j) = z_i * z_j
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        
    def decode_all(self, z):
        adj_logits = torch.matmul(z, z.t())
        return torch.sigmoid(adj_logits)
        
    def recon_loss(self, z, pos_edge_index, neg_edge_index):
        """
        Calculates BCE loss on positive and negative edges directly.
        """
        pos_logits = self.decode(z, pos_edge_index)
        pos_loss = -F.logsigmoid(pos_logits).mean()

        if neg_edge_index.size(1) > 0:
            neg_logits = self.decode(z, neg_edge_index)
            # log(1 - sigmoid(x)) = logsigmoid(-x)
            neg_loss = -F.logsigmoid(-neg_logits).mean()
        else:
            neg_loss = 0.0
            
        return pos_loss + neg_loss
