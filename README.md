# The Object

The object of this repository is to tentatively examine if cosine similarity between two words is at all predictive of the semantic priming response in humans.

# The Architecture

The project *should* be as simple as defining a general class GutsGorer, which wraps a HuggingFace transformer and generates the embedding for a particurly input. We'll start with the assumption that the user will provide the GutsGorer with words. In this case, when there is a multi-token word provided, we'll take the mean vector across the tokens.

