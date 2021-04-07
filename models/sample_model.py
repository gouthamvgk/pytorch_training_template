import torch
from torch import nn

class my_model(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.conifg = config
		#Define all yours layers below

	def forward(self, input):
		#Forward pass through layers here
		out = self.layers(input)
		return out