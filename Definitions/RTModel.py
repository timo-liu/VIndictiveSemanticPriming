from torch import nn
from typing import List, Tuple
from torch.nn import Linear

class RTModel(nn.Module):
	def __init__(self,
				 architecture : List[Tuple[int, int]],
				 activation : type[nn.Module]
				 ):
		super().__init__()
		self.architecture = architecture
		self.activation = activation()
		self.model = nn.ModuleList([Linear(i, o) for i,o in architecture])
	def forward(self, x):
		for l in self.model[:-1]:
			x = l(x)
			x = self.activation(x)
		x = self.model[-1](x)
		return x