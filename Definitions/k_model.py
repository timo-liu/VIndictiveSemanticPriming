import torch.nn as nn
from typing import List

class k_model(nn.Module):
	def __init__(self, architecture : List[int] = [768, 192, 48, 12]):
		super().__init__()
		self.mlp = nn.ModuleList([nn.Linear(architecture[i], architecture[i + 1]) for i in range(len(architecture) - 1)])
		self.activation = nn.ReLU()

	def forward(self, x):
		for layer in self.mlp:
			x = layer(x)
			x = self.activation(x)
		return x