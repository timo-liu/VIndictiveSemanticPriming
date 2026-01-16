import pickle
from sklearn.neighbors import KDTree
from typing import List, Dict
import numpy as np
import os
from Definitions.GutsGorer import GutsGorer

class ToothPuller:
	def __init__(self, word_list : List,
			   reference_word_list : List,
			   data_dir : str,
			   hf_id : str = "bert-base-uncased"):
		self.word_list = word_list
		self.reference_word_list = reference_word_list
		self.data_dir = data_dir
		self.reference_embeddings = []
		self.gg = GutsGorer(hf_id=hf_id)

# region utils

	def get_output_path(self, component : str = ""):
		output_path = os.path.join(self.data_dir, f"precomputed_embeddings_{component}.pkl")
		return output_path
	def check_reference_word_list_file(self, component : str = ""):
		if os.path.exists(self.get_output_path(component)):
			return True
		else:
			return False

# endregion utils

	def embed_reference_list(self, component: str = "") -> Dict[str, np.ndarray]:
		assert len(self.reference_word_list) > 0, "Reference word list must be greater than 0"

		output_path = self.get_output_path(component)

		# Load existing data if it exists
		if os.path.exists(output_path):
			print(f"Loading existing reference embeddings from {output_path}")
			with open(output_path, "rb") as f:
				data: Dict[str, np.ndarray] = pickle.load(f)
		else:
			data = {}

		# Compute embeddings only for missing words
		for word in self.reference_word_list:
			word = self.gg.preprocess(word)
			if word not in data:
				data[word] = self.gg.compute_embedding(word, component)

		# Update internal state
		self.reference_embeddings = list(data.values())

		# Save updated data
		with open(output_path, "wb") as f:
			pickle.dump(data, f)

		return data