How to work with Open-Source LLMs: resources for social scientists (working draft)
==============

**Last updated:** *Oct-14 2023*

**Authors:** *Marie-Lou Laprise and Angela Li*

# Useful prerequisites

* [Practical Deep Learning for Coders](https://course.fast.ai/)

# Obtaining pre-trained weights: Hugging Face 101

Useful resources:
* [Hugging Face Transformers Library (Python library documentation)](https://huggingface.co/docs/transformers/philosophy), built by and for the open-source community
  * Hugging Face is a great way to get started with LLMs. For most social science use cases, it probably represents the ideal trade-off between ease of use and flexibility
* [Model Hub](https://huggingface.co/models)
  * Start with a model from this list
* [What LLM to use?](https://github.com/continuedev/what-llm-to-use)
  * An overview of both open-source and commercial options, up to date October 2023
* To go further: [Complete Transformers Course](https://huggingface.co/learn/nlp-course/chapter1/1) and its [Github repo](https://github.com/huggingface/course)

Why pre-trained?
* See white paper on [Current Best Practices for Training LLMs from Scratch](https://files.catbox.moe/6x8ct9.pdf) by Rebecca Li, Andrea Parker, Justin Tenuto

## Three classes:
HF simplifies the pipeline for using any model in its library to three classes of objects:

1. **Models**: This object stores the pre-trained weights. For us, it will be a PyTorch model, `torch.nn.Module`.
2. **Configurations**: This object stores the hyperparameters (number of layers, size of layers, etc.). You only need to set it manually if you intend to do some training or fine-tuning.
3. Input adapted **pre-processing**: This object maps the raw training data into a processed version that can be used for training or fine-tuning. For LLMs, this is a tokenizer and it stores the vocabulary and methods for encoding strings (making them machine-readable) and decoding strings (making them human-readable).

## Three methods:

### (1) Loading:
The first step is to select a "pretrained checkpoint" from the HF Hub (or a locally saved checkpoint). The `from_pretrained()` method initializes all three required classes by downloading, caching, and loading class instances and related data. Such data includes, for instance, hyperparameters, vocabulary, and weights.

### (2) Saving and (3) publishing:
If you make modifications to either of the three classes (model, configuration, tokenizer), you can save a checkpoint locally with the `to_pretrained()` method, for subsequent retrieval with `from_pretrained()`.  If you want to publish the checkpoint to share it with others, you can use the `push_to_hub()` method instead.

## Two APIs:
There are two main APIs: an upstream `Trainer` to train or fine-tune PyTorch models and a downstream `pipeline()` for inference. This is modular so different frameworks (PyTorch or other) can be used for training and inference, and we can easily switch the model, parameter, or pre-processing that we are using.

# Fine-tuning or Low-Rank Adaptation (LoRA)
## Practical guides
Useful resources:
* [The Novice's LLM Training Guide (blog)](https://rentry.co/llm-training) by [Alpin Dale](https://github.com/AlpinDale)
  * A step-by-step guide to fine-tuning, LoRA, QLoRA, and more.
* [Transformer Math 101 (blog)](https://blog.eleuther.ai/transformer-math/) by Quentin Anthony, Stella Biderman, Hailey Schoelkopf
  * Useful rules of thumb to estimate computation and memory usage for various transformers tasks
  * See also [Model training anatomy (HF docs)](https://huggingface.co/docs/transformers/model_memory_anatomy) for a more detailed explanation of memory use during training
* [Llama Factory (Github)](https://github.com/hiyouga/LLaMA-Factory): code, json datasets, and other resources to help fine-tune some LLMs; supports Llama and Llama-2
* [Llama 2 Fine-tuning / Inference Recipes and Examples (Github)](https://github.com/facebookresearch/llama-recipes/) and [Hugging Face's Transformers Examples (Github)](https://github.com/huggingface/transformers/tree/main/examples)
  * Avoid reinventing the wheel by first looking for similar examples in the many scripts shared in those two repos
* [Tips for Working with HF on Princeton's Research Computing Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face)
  * Avoid memory and storage issues on the HPC cluster
* [Non-engineers guide: Train a LLaMA 2 chatbot (HF blog)](https://huggingface.co/blog/Llama2-for-non-engineers) by Andrew Jardine and Abhishek Thakur
  * How to fine-tune a LlaMa model for chat without writing any code (this is more of an "at-home" version for those without access to high-performance computing and/or without coding experience)

For models with billions of parameters, fine-tuning requires hundreds of GBs of VRAM. For a 7B model, fine-tuning requires up to ~200GB of memory. You should have at least 100MB of high-quality fine-tuning data.

Steps to process data for fine-tuning: parse, cluster, filter, prepare for tokenization, create train/validation splits. If training data too large, convert to Apache Arrow. Processing will vary depending on choice of fine-tuning method (supervised, unsupervised, RLHF, etc.)

It is generally much easier to use a single GPU if hardware requirements allow it, because of complications caused by model parallelism.

Memory can be reduced to 1/3 of the initial requirement by using LoRA:

* See [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (article)
* See [PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft) (HF blog)

If that is still untractable, it can be further reduced with Quantized LoRA:

* This relies on the `bitsandbytes` Python library
* More about quantization here: 
  * [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) and [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (articles) 
  * [LLM.int8() and Emergent Features](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) (blog about theory) by Tim Dettmers
  * [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration) (blog about application) by Younes Belkada and Tim Dettmers

With QLoRA, it becomes possible to fine-tune a 65B parameter model on a single 48GB GPU.

## Researchers choices in fine-tuning paradigms and methods

* Reinforcement learning with human feedback: [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft) (HF blog)
* Regularization: [NEFTUNE: Noisy Embeddings Improve Finetuning](https://arxiv.org/pdf/2310.05914.pdf) (Jain et al. 2023) (Pre-print)

# Inference

Useful resources:
* [Llama 2 Fine-tuning / Inference Recipes and Examples (Github)](https://github.com/facebookresearch/llama-recipes/)
* [Tips for Working with HF on Princeton's Research Computing Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face)

To go further:
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (blog)

If you need to run the model locally on CPU only:
* [Llama.cpp](https://github.com/ggerganov/llama.cpp) for fast / efficient inference locally on a MacBook, command line interface
* [Oobabooga](https://github.com/oobabooga/text-generation-webui) for a web-based user interface to Transformers or Llama.cpp

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

# What about non-language tasks?

* [Probabilistic Time Series Forecasting with 🤗 Transformers](https://huggingface.co/blog/time-series-transformers) by Niels Rogge and Kashif Rasul