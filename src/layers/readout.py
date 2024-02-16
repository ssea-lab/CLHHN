import torch
import torch.nn as nn


class LastAttention(nn.Module):
	def __init__(self, hidden_dim):
		super(LastAttention, self).__init__()
		self.hidden_size = hidden_dim
		self.pos_size = hidden_dim / 2
		self.position_embedding = nn.Embedding(100, self.pos_size)
		self.w_hp = nn.Linear(self.hidden_size + self.pos_size, self.hidden_size, bias=False)
		self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
		self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
		self.q = nn.Linear(self.hidden_size, 1, bias=False)

	def forward(self, seq_hidden, mask):
		# position embedding only for calculate attn scores, not in part of the final seq embedding
		batch_size, seq_len = mask.shape
		mask = mask.float().unsqueeze(-1)

		pos_emb = self.position_embedding.weight[:seq_len]
		pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

		h_t = seq_hidden[:, 0]                              # (b, h)
		h_t = h_t.unsqueeze(1).repeat(1, seq_len, 1)        # (b, l, h)

		z = torch.tanh(self.w_hp(torch.cat([seq_hidden, pos_emb], dim=-1)))     # (b, l, h)
		beta = self.q(torch.sigmoid(self.w1(z) + self.w2(h_t)))                 # (b, l, 1)
		seq_output = torch.sum(beta * seq_hidden * mask, 1)                     # (b, h)

		return seq_output



