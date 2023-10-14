How to work with Open-Source LLMs: resources for social scientists
==============

**Last updated:** *Oct-14 2023*
**Authors:** *Marie-Lou Laprise and Angela Li*

# Useful prerequisites

* [Practical Deep Learning for Coders](https://course.fast.ai/)

# Obtaining pre-trained weights: Hugging Face 101

Useful resource:
* [Hugging Face Transformers Library (Python library documentation)](https://huggingface.co/docs/transformers/philosophy), built by and for the open-source community
  * Hugging Face is a great way to get started with LLMs. For most social science use cases, it probably represents the ideal trade-off between ease of use and flexibility
* [Model Hub](https://huggingface.co/models)
  * Start with a model from this list
* To go further: [Complete Transformers Course](https://huggingface.co/learn/nlp-course/chapter1/1) and its [Github repo](https://github.com/huggingface/course)

## Three classes:
HF simplifies the pipeline for using any model in its library to three classes of objects:

1. Models: This object stores the pre-trained weights. For us, it will be a PyTorch model, `torch.nn.Module`.
2. Configurations: This object stores the hyperparameters (number of layers, size of layers, etc.). You only need to set it manually if you intend to do some training or fine-tuning.
3. Input adapted pre-processing: This object maps the raw training data into a processed version that can be used for training or fine-tuning. For LLMs, this is a tokenizer and it stores the vocabulary and methods for encoding strings (making them machine-readable) and decoding strings (making them human-readable).

## Three methods:

### (1) Loading:
The first step is to select a "pretrained checkpoint" from the HF Hub (or a locally saved checkpoint). The `from_pretrained()` method initializes all three required classess by downloading, caching, and loading class instances and related data. Such data includes, for instance, hyperparameters, vocabulary, and weights.

### (2) Saving and (3) publishing:
If you make modifications to either of the three classes (model, configuration, tokenizer), you can save a checkpoint locally with the `to_pretrained()` method, for subsequent retrieval with `from_pretrained()`.  If you want to publish the checkpoint to share it with others, you can use the `push_to_hub()` method instead.

## Two APIs:
There are two main APIs: an upstream `Trainer` to train or fine-tune PyTorch models and a downstream `pipeline()` for inference. This is modular so different frameworks (PyTorch or other) can be used for training and inference, and we can easily switch the model, parameter, or pre-processing that we are using.

# Fine-tuning

Useful resources:
* [The Novice's LLM Training Guide (blog)](https://rentry.co/llm-training) by [Alpin Dale](https://github.com/AlpinDale)
  * A step-by-step guide to fine-tuning, LoRA, QLoRA, and more.
* [Non-engineers guide: Train a LLaMA 2 chatbot (HF blog)](https://huggingface.co/blog/Llama2-for-non-engineers) by Andrew Jardine and Abhishek Thakur
  * How to fine-tune a LlaMa model for chat without writing any code (this is more of an "at-home" version for those without access to high-performance computing and/or without coding experience)
* [Transformer Math 101 (blog)](https://blog.eleuther.ai/transformer-math/) by Quentin Anthony, Stella Biderman, Hailey Schoelkopf
  * Useful rules of thumb to estimate computation and memory usage for various transformers tasks
  * See also [Model training anatomy (HF docs)](https://huggingface.co/docs/transformers/model_memory_anatomy) for a more detailed explanation of memory use during training
* [Llama 2 Fine-tuning / Inference Recipes and Examples (Github)](https://github.com/facebookresearch/llama-recipes/) and [Hugging Face's Transformers Examples (Github)](https://github.com/huggingface/transformers/tree/main/examples)
  * Avoid reinventing the wheel by first looking for similar examples in the many scripts shared in those two repos
* [Tips for Working with HF on Princeton's Research Computing Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face)
  * Avoid memory and storage issues on the HPC cluster

# Inference

Useful resources:
* [Llama 2 Fine-tuning / Inference Recipes and Examples (Github)](https://github.com/facebookresearch/llama-recipes/)
* [Tips for Working with HF on Princeton's Research Computing Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face)


# Incorporating LLMs in downstream tasks

In R:

* [Deep Learning and Scientific Computing with R `torch` (ebook)](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/) by Sigrid Keydana
  * How to use the R interface to `PyTorch`
* The [`chattr` package](https://mlverse.github.io/chattr/articles/other-interfaces.html) lets you interact with local models in R scripts -- using the `chattr()` function -- and integrate inference outputs in your research pipeline.

In Python:

* Build applications with [LangChain](https://github.com/langchain-ai/langchain)
  * Once you have a model ready for inference, LangChain can help you build applications for it, including sequences of calls or AI agents.
  * Sometimes described as a "prompt-orchestration" tool, it helps coordinate series of (potentially inter-dependent) small tasks with context and memory.
* The [`spaCy` library](https://spacy.io/usage/spacy-101) allows you to use local models for various NLP tasks.
  * The LLM integration is relatively new. See the [documentation](https://spacy.io/usage/large-language-models)
  * Useful library extensions include [SpanMarker for named entity recognition](https://tomaarsen.github.io/SpanMarkerNER/notebooks/spacy_integration.html)