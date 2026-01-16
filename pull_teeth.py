"""
tooth_puller.py
Uses GutsGorer to extract cosine similarities between words and returns top matches

Word list source: https://github.com/dwyl/english-words
"""
from scipy.spatial import KDTree
import json
from Definitions.ToothPuller import ToothPuller
import os
import argparse

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('input_file', type=str, help='txt of words to find targets for')
	arg_parser.add_argument('reference_word_file', type=str, help="txt of words within which a target will be found")
	arg_parser.add_argument('-i', '--hf_id', required=False, type=str, help='hf id', default="bert-base-uncased")
	arg_parser.add_argument('-s', '--save', required=False, type=str, help='output dir', default="dentist_tray")
	arg_parser.add_argument('-c', '--component', required=False, type=str, help='component of interest', default="")
	arg_parser.add_argument('--cache', required=False, type=str, help='cache dir', default="cache")
	arg_parser.add_argument('--n_ref', required=False, type=int, help='number of words to process from the seking, reference maw', default=200)
	arg_parser.add_argument('-k', '--top_k', required=False, type=int, help='number of targets to get', default=10)
	args = arg_parser.parse_args()

	if not os.path.exists(args.cache):
		os.makedirs(args.cache)

	if os.path.exists(args.input_file) and os.path.exists(args.reference_word_file):
		with open(args.reference_word_file, "r") as refs, open(args.input_file, "r") as words, open(args.input_file, "r") as input:
			input_list = [x.strip() for x in input.readlines()]
			ref_list = [x.strip() for x in refs.readlines()[:args.n_ref]]
			word_list = [x.strip() for x in words.readlines()]
	tp = ToothPuller(
		word_list,
		ref_list,
		args.cache,
		args.hf_id
	)

	print("Computing embeddings for the reference words...")
	data = tp.embed_reference_list(args.component)
	words = list(data.keys())
	embeds = list(data.values())
	embeds = [ emb / sum(emb ** 2) ** 0.5 for emb in embeds]

	print("Building KDTree...")
	indexer = KDTree(embeds)
	print("Built KDTree")

	output_file = os.path.join(args.save, f"neighbors_generated_by_{args.hf_id}_{args.component}.json")

	with open(output_file, "w") as out:
		for word in input_list:
			neighbors = []
			distances = []
			query_vector = tp.gg.compute_embedding(word, args.component)
			normalized_query = query_vector / sum(query_vector ** 2) ** 0.5
			di, idx = indexer.query(normalized_query, k=args.top_k + 1)
			for i, index in enumerate(idx.ravel()):
				neighbors.append(words[index])
				distances.append(di.ravel()[i])

				# if same word, then break, but otherwise oh well.

				if len(distances) >= args.top_k:
					break
			dump = {
				"word" : word,
				"neighbors" : neighbors,
				"distances" : distances
			}

			out.write(json.dumps(dump) + '\n')