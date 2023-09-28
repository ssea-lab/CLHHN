import torch
from torch import nn


# Weighted Lossless HyperGraph Attention Convolutional Neural Network
class WLHGAConvWeight(nn.Module):
	def __init__(self, hidden_dim, step, dropout=0.5):
		super(WLHGAConvWeight, self).__init__()
		self.hidden_size = hidden_dim
		self.step = step
		self.dropout = nn.Dropout(dropout)

		self.q1 = nn.Linear(self.hidden_size + 1, 1, bias=False)
		self.q2 = nn.Linear(self.hidden_size + 1, 1, bias=False)
		self.leakyrelu = nn.LeakyReLU(0.2)

	def agg_cell(self, nodes_hidden, edge_hidden, hn_adj):
		batch_size, edge_num, node_num = hn_adj.shape
		# ============= node to edge
		edge_hidden_att = edge_hidden.unsqueeze(2).repeat(1, 1, node_num, 1)    # (b, edge_num, node_num, h)
		nodes_hidden_att = nodes_hidden.unsqueeze(1).repeat(1, edge_num, 1, 1)
		ele_hidden = torch.cat([edge_hidden_att * nodes_hidden_att, self.dropout(hn_adj.unsqueeze(-1))], dim=-1)
		alpha = self.leakyrelu(self.q1(ele_hidden)).squeeze(-1)                 # (b, edge_num, node_num)
		alpha.masked_fill_(hn_adj == 0, -1e10)
		alpha = torch.softmax(alpha, dim=-1)
		edge_hidden_new = torch.matmul(alpha, nodes_hidden)         # (b, en, nn) @ (b, nn, h) -> (b, en, h)
		# ============= edge to node
		hn_adj_t = hn_adj.transpose(1, 2)
		edge_hidden_attn = edge_hidden_new.unsqueeze(1).repeat(1, node_num, 1, 1)
		nodes_hidden_attn = nodes_hidden.unsqueeze(2).repeat(1, 1, edge_num, 1)
		ele_hidden = torch.cat([edge_hidden_attn * nodes_hidden_attn, self.dropout(hn_adj_t.unsqueeze(-1))], dim=-1)
		beta = self.leakyrelu(self.q2(ele_hidden)).squeeze(-1)      # (b, nn, en)
		beta.masked_fill_(hn_adj_t == 0, -1e10)
		beta = torch.softmax(beta, dim=-1)
		nodes_hidden_new = torch.matmul(beta, edge_hidden_new)      # (b, nn, en) @ (b, en, h) -> (b, nn, h)

		return nodes_hidden_new, edge_hidden_new

	def forward(self, nodes_hidden, hn_adj):
		# nodes_hidden: (b, node_num, h) | hn_adj: (b, edge_num, node_num)

		# ============= initial hyperEdge hidden representation (weighted)
		edge_nodes_num = torch.sum(hn_adj, dim=-1).unsqueeze(-1)    # (b, edge_num, 1)
		edge_nodes_num = torch.where(edge_nodes_num == 0., torch.ones_like(edge_nodes_num), edge_nodes_num)
		edge_hidden = torch.matmul(hn_adj, nodes_hidden)            # (b, edge_num, h)
		edge_hidden = edge_hidden / edge_nodes_num                  # (b, edge_num, h)
		for i in range(self.step):
			nodes_hidden, edge_hidden = self.agg_cell(nodes_hidden, edge_hidden, hn_adj)
		return nodes_hidden, edge_hidden
