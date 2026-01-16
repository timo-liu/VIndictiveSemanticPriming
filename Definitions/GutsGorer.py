from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor, tensor
from torch.nn.functional import cosine_similarity as _cosine_similarity
import numpy as np
from typing import Union

"""
Likely breakdown of categories
{
	1.0 : first-associate related.
	2.0 : first-associate unrelated,
	3.0 : other-associate related,
	4.0 : other-associate unrelated
}
"""

def get_encoder_layers(model):
	if hasattr(model, "encoder"):
		return model.encoder.layer
	elif hasattr(model, "transformer"):
		return model.transformer.layer
	elif model.__class__.__name__.lower().startswith("albert"):
		return model.encoder.albert_layer_groups
	else:
		raise ValueError(f"Unsupported model type: {type(model)}")


class GutsGorer:
	def __init__(self, hf_id : str):
		self.name = hf_id

		self.model = self.load_model(hf_id)
		self.tokenizer = self.load_tokenizer(hf_id)
		
	# region loading_utilities
	def load_model(self, hf_id : str) -> AutoModel:
		"""
		Loads the huggingface transformer model
		:param hf_id:
		:return:
		"""
		try:
			model = AutoModel.from_pretrained(hf_id, output_hidden_states=True)
		except Exception as e:
			print(e)
			return None
		return model

	def load_tokenizer(self, hf_id : str) -> AutoTokenizer:
		"""
		Loads the huggingface tokenizer
		:param hf_id:
		:return:
		"""
		try:
			tokenizer = AutoTokenizer.from_pretrained(hf_id)
		except Exception as e:
			print(e)
			return None
		return tokenizer

	# endregion loading_utilities

	# region embedding_utilities
	def cosine_similarity(self, vector1 : Tensor, vector2 : Tensor) -> float:
		"""
		Compute the cosine similarity between two vectors
		:param vector1:
		:param vector2:
		:return:
		"""
		return _cosine_similarity(vector1, vector2, dim=0).item()


	# endregion embedding_utilities

	def compute_embedding(self, word : str, component : str = "", return_all : bool = False) -> Union[Tensor, List[Tensor]]:
		"""
		Compute the embedding vector for a given word using a specified model component.

		The input word is preprocessed by lowercasing, stripping whitespace,
		and prepending a space (to handle most tokenizers correctly). The word
		is then tokenized.

		- Single-token words return a single embedding vector.
		- Multi-token words return the **mean vector** of all token embeddings.

		Embeddings can be extracted from different components:
		- `"word_embeddings"`: The raw word embeddings from the model’s embedding layer.
		- `"encoder_layer_X"`: Hidden states from the encoder layer X (0-indexed).
		  Example: `"encoder_layer_5"` extracts the output of encoder layer 5.

		Parameters
		----------
		word : str
			The input word to compute the embedding for.
		component : str, optional
			The model component to extract embeddings from. Must be either
			`"word_embeddings"` or `"encoder_layer_X"`. If empty or invalid,
			an AssertionError is raised. Default is `""`.

		Returns
		-------
		torch.Tensor
			A numpy array of shape `(hidden_size,)` representing the embedding
			of the word. Multi-token words are averaged over tokens.

		Raises
		------
		AssertionError
			If `component` is empty or does not match a supported component.
			If `encoder_layer_X` is not a valid integer layer id.
		"""
		# preprocessing
		word = self.preprocess(word)
		encoded = self.tokenizer(
			word,
			return_tensors="pt",
			add_special_tokens=True
		)
		input_ids = encoded["input_ids"][0]

		# remove special tokens safely
		special_ids = set(self.tokenizer.all_special_ids)
		word_id = [i for i in input_ids.tolist() if i not in special_ids]

		# if len(word_id > 1), then mean, otherwise return
		# getting the embeddings from the vocab index
		if component == "word_embeddings":
			with torch.no_grad():
				# Forward through embeddings only
				input_tensor = torch.tensor(word_id).unsqueeze(0)
				result = self.model.get_input_embeddings()(input_tensor)
				embeddings = result.squeeze(0).cpu().detach().numpy()
				return np.mean(embeddings, axis=0)
		elif component.startswith("encoder_layer_"):
			layer_id_str = component.replace("encoder_layer_", "")
			assert layer_id_str.isdigit(), f"Invalid encoder layer id: {layer_id_str}"
			layer_id = int(layer_id_str)
			layers = get_encoder_layers(self.model)
			assert layer_id < len(layers), f"Invalid encoder layer id: {layer_id_str}"

			with torch.no_grad():
				# Forward through embeddings only
				input_tensor = torch.tensor(word_id).unsqueeze(0)
				result = self.model(input_tensor)
				hidden_states = result.hidden_states
				if not return_all:
					embeddings = hidden_states[layer_id].squeeze(0).cpu().detach().numpy()
				else:
					embeddings = hidden_states
			if not return_all:
				return np.mean(embeddings, axis=0)
			else:
				return [np.mean(emb.squeeze(0).cpu().detach().numpy(), axis=0) for emb in embeddings]
		elif component.startswith("decoder_layer_"):
			assert False, "Uhoh"
		else:
			assert component == "", f"{component} is either not implemented for extraction, or does not exist in the given model."
			with torch.no_grad():
				input_tensor = torch.tensor(word_id).unsqueeze(0)
				result = self.model(input_tensor)
				embeddings = result.last_hidden_state[0].cpu().numpy()
				return np.mean(embeddings, axis=0)

	def cosine_compare(self, word1 : str, word2 : str, component : str) -> float:
		embedding1 = torch.from_numpy(self.compute_embedding(word1, component= component))
		embedding2 = torch.from_numpy(self.compute_embedding(word2, component= component))
		return self.cosine_similarity(embedding1, embedding2)

	def preprocess(self, word : str) -> str:
		return word.lower().strip()
		# return ' ' + word.lower().strip()