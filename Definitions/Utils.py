from typing import Dict, Optional
import pandas as pd
import os
import numpy as np

def parse_data(path : str,
			   extract_dict : Optional[Dict[str, str]] = None):
	"""
	Load a CSV or Excel file and extract selected columns into a new DataFrame.

	The function reads a dataset from the given file path and constructs a new
	pandas DataFrame containing columns specified in `extract_dict`. Each key
	in `extract_dict` becomes a column name in the returned DataFrame, while
	the corresponding value specifies the column name to extract from the
	source DataFrame.

	If a specified source column is not found, a warning message is printed
	and the column is skipped.

	Parameters
	----------
	path : str
		Path to the input data file. Supported formats are `.csv` and `.xlsx`.
	extract_dict : Dict[str, str], optional
		A mapping from output column names to source column names in the input
		DataFrame. Defaults to extracting prime, target, primecondition, RT,
		and accuracy-related columns.

	Returns
	-------
	pandas.DataFrame
		A DataFrame containing the extracted columns with renamed headers.

	Raises
	------
	ValueError
		If the file extension is not `.csv` or `.xlsx`.

	Examples
	--------
	>>> df = parse_date("experiment_data.csv")
	>>> df.columns
	Index(['prime', 'target', 'primecondition', 'RT', 'accuracy'], dtype='object')
	"""

	if extract_dict is None:
		extract_dict = {
				   "prime" : "prime",
				   "target" : "target",
				   "primecondition" : "primecond",
				   "RT" : "target.RT",
				   "accuracy" : "target.ACC",
					"isi" : "isi"
			   }

	if path.endswith('.xlsx'):
		df = pd.read_excel(path)
	elif path.endswith('.csv'):
		df = pd.read_csv(path)

	# construct the return df
	rdf = pd.DataFrame()
	for title, target_column in extract_dict.items():
		try:
			rdf[title] = df[target_column]
		except KeyError:
			print(f"{target_column} not found in {path}")

	return rdf

def load_embedding_cache(cache_path):
	if os.path.exists(cache_path):
		return dict(np.load(cache_path, allow_pickle=True))
	return {}

def save_embedding_cache(cache, cache_path):
	os.makedirs(os.path.dirname(cache_path), exist_ok=True)
	np.savez_compressed(cache_path, **cache)

def get_or_compute_encoder_layers(word, gg, cache, component, *, normalize=False):
	"""
	Return encoder-layer embeddings for a word.
	Shape: (n_layers, hidden_dim)

	NOTE:
	- Does NOT handle word_embeddings
	"""
	if word in cache:
		return cache[word]

	layer_embs = gg.compute_embedding(
		word,
		component=component,
		return_all=True
	)

	# stack → (layers, dim)
	embs = np.stack(layer_embs, axis=0)

	if normalize:
		norms = np.linalg.norm(embs, axis=1, keepdims=True)
		embs = embs / np.clip(norms, 1e-9, None)

	cache[word] = embs
	return embs
