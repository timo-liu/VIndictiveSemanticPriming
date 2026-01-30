from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity as _cosine_similarity
import numpy as np
from typing import Union, List


def get_encoder_layers(model):
	if model.__class__.__name__.lower().startswith("albert"):
		return model.encoder.albert_layer_groups
	elif hasattr(model, "encoder"):
		return model.encoder.layer
	elif hasattr(model, "transformer"):
		return model.transformer.layer
	else:
		raise ValueError(f"Unsupported model type: {type(model)}")


class GutsGorer:
	def __init__(self, hf_id: str):
		self.name = hf_id

		# ------------------------------
		# Device selection
		# ------------------------------
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model = self.load_model(hf_id)
		self.tokenizer = self.load_tokenizer(hf_id)

		if self.model is not None:
			self.model.to(self.device)
			self.model.eval()

	# ------------------------------------------------------------------
	# Loading utilities
	# ------------------------------------------------------------------

	def load_model(self, hf_id: str) -> AutoModel:
		try:
			model = AutoModel.from_pretrained(
				hf_id,
				output_hidden_states=True
			)
		except Exception as e:
			print(e)
			return None
		return model

	def load_tokenizer(self, hf_id: str) -> AutoTokenizer:
		try:
			tokenizer = AutoTokenizer.from_pretrained(hf_id)
		except Exception as e:
			print(e)
			return None
		return tokenizer

	# ------------------------------------------------------------------
	# Embedding utilities
	# ------------------------------------------------------------------

	def cosine_similarity(self, vector1: Tensor, vector2: Tensor) -> float:
		return _cosine_similarity(vector1, vector2, dim=0).item()

	def preprocess(self, word: str) -> str:
		return word.lower().strip()

	def compute_embedding(
		self,
		word: str,
		component: str = "",
		return_all: bool = False
	) -> Union[np.ndarray, List[np.ndarray]]:
		"""
		Compute the embedding vector for a given word using a specified model component.
		GPU-enabled; outputs are always returned as NumPy arrays.
		"""

		word = self.preprocess(word)

		encoded = self.tokenizer(
			word,
			return_tensors="pt",
			add_special_tokens=True
		)

		input_ids = encoded["input_ids"][0]

		# Remove special tokens safely
		special_ids = set(self.tokenizer.all_special_ids)
		word_id = [i for i in input_ids.tolist() if i not in special_ids]

		input_tensor = torch.tensor(word_id, device=self.device).unsqueeze(0)

		with torch.no_grad():
			if component == "word_embeddings":
				result = self.model.get_input_embeddings()(input_tensor)
				embeddings = result.squeeze(0)
				return embeddings.mean(dim=0).cpu().numpy()

			elif component.startswith("encoder_layer_"):
				layer_id_str = component.replace("encoder_layer_", "")
				# assert layer_id_str.isdigit(), f"Invalid encoder layer id: {layer_id_str}"
				layer_id = int(layer_id_str)

				layers = get_encoder_layers(self.model)
				# assert layer_id < len(layers), f"Invalid encoder layer id: {layer_id_str}"

				result = self.model(input_tensor)
				hidden_states = result.hidden_states

				if not return_all:
					layer_embeddings = hidden_states[layer_id].squeeze(0)
					return layer_embeddings.mean(dim=0).cpu().numpy()
				else:
					return [
						h.squeeze(0).mean(dim=0).cpu().numpy()
						for h in hidden_states
					]

			elif component.startswith("decoder_layer_"):
				raise AssertionError("Decoder layers are not supported.")

			else:
				assert component == "", (
					f"{component} is either not implemented for extraction "
					"or does not exist in the given model."
				)

				result = self.model(input_tensor)
				embeddings = result.last_hidden_state.squeeze(0)
				return embeddings.mean(dim=0).cpu().numpy()

	def cosine_compare(self, word1: str, word2: str, component: str) -> float:
		embedding1 = torch.from_numpy(
			self.compute_embedding(word1, component=component)
		)
		embedding2 = torch.from_numpy(
			self.compute_embedding(word2, component=component)
		)
		return self.cosine_similarity(embedding1, embedding2)
