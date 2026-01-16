import wandb
import torch.nn as nn
import torch
from typing import List, Tuple
from Definitions.RTModel import RTModel
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

class RTPredictor:
	def __init__(self,
				 architecture : List[Tuple[int, int]],
				 activation : type[nn.Module],
				 criterion : type[nn.Module]):
		self.model = RTModel(architecture, activation)
		self.criterion = criterion()
		self.optimizer = Adam(self.model.parameters(), lr=0.001)

	def train(
			self,
			xtr,
			ytr,
			xval,
			yval,
			epochs: int,
			patience: int,
			project_name: str,
			verbose: bool = True,
			batch_size: int = 32,
			eval_every: int = 1
	):
		best_val_loss = float('inf')
		current_patience = patience

		wandb.init(project=project_name)

		# ===== DATA LOADERS =====
		train_loader = DataLoader(
			TensorDataset(xtr, ytr),
			batch_size=batch_size,
			shuffle=True
		)

		val_loader = DataLoader(
			TensorDataset(xval, yval),
			batch_size=batch_size,
			shuffle=False
		)

		for epoch in range(epochs):
			# ===== TRAIN =====
			self.model.train()
			train_loss = 0.0

			for xb, yb in train_loader:
				self.optimizer.zero_grad()

				preds = self.model(xb)
				loss = self.criterion(preds, yb)  # MSE

				loss.backward()
				self.optimizer.step()

				train_loss += loss.item() * xb.size(0)

			train_loss /= len(train_loader.dataset)

			wandb.log({'train_loss': train_loss, 'epoch': epoch + 1})

			if verbose:
				print(f"Epoch ({epoch + 1}/{epochs}) | Train MSE: {train_loss:.6f}")

			# ===== VALIDATION =====
			if (epoch + 1) % eval_every == 0:
				self.model.eval()
				val_loss = 0.0

				with torch.no_grad():
					for xb, yb in val_loader:
						preds = self.model(xb)
						loss = self.criterion(preds, yb)
						val_loss += loss.item() * xb.size(0)

				val_loss /= len(val_loader.dataset)

				wandb.log({'val_loss': val_loss, 'epoch': epoch + 1})

				if verbose:
					print(f"Epoch ({epoch + 1}/{epochs}) | Val MSE: {val_loss:.6f}")

				# ===== EARLY STOPPING =====
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					current_patience = patience

					best_state = {
						'model': self.model.state_dict(),
						'optimizer': self.optimizer.state_dict(),
						'epoch': epoch,
						'val_loss': best_val_loss
					}
				else:
					current_patience -= 1
					if current_patience == 0:
						if verbose:
							print("Early stopping triggered.")
						break

		# ===== RESTORE BEST MODEL =====
		if 'best_state' in locals():
			self.model.load_state_dict(best_state['model'])

	def test_model(self, xtest, ytest, batch_size: int = 32, return_preds=False):
		self.model.eval()

		test_loader = DataLoader(
			TensorDataset(xtest, ytest),
			batch_size=batch_size,
			shuffle=False
		)

		test_loss = 0.0
		all_preds = []

		with torch.no_grad():
			for xb, yb in test_loader:
				preds = self.model(xb)
				loss = self.criterion(preds, yb)
				test_loss += loss.item() * xb.size(0)

				if return_preds:
					all_preds.append(preds.cpu())

		test_loss /= len(test_loader.dataset)

		if return_preds:
			return test_loss, torch.cat(all_preds, dim=0)
		return test_loss