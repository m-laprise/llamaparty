How to work with Open-Source LLMs: resources for social scientists (working draft)
==============

**Last updated:** *Oct-14 2023*

**Authors:** *Marie-Lou Laprise and Angela Li*

# Useful prerequisites

* [Practical Deep Learning for Coders (free course)](https://course.fast.ai/) by fast.ai and Jeremy Howard
* [Understanding Large Language Models: A Cross-Section of the Most Relevant Literature To Get Up to Speed](https://magazine.sebastianraschka.com/p/understanding-large-language-models) by Sebastian Raschka
* [REFORMS: Reporting Standards for Machine Learning Based Science (pre-print)](https://arxiv.org/abs/2308.07832)

# Obtaining pre-trained weights: Hugging Face 101

Useful resources:
* [Hugging Face Transformers Library (Python library documentation)](https://huggingface.co/docs/transformers/philosophy), built by and for the open-source community
  * Hugging Face is a great way to get started with LLMs. For most social science use cases, it probably represents the ideal trade-off between ease of use and flexibility
* [Model Hub](https://huggingface.co/models)
  * Start with a model from this list
* Choosing a model:
  * This 2023 review paper [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712) includes a discussion of available foundation model families and architectures, both commercial and open-source, with their respective strengths and limitations, along with a great family tree visualization
  * The [Transformer models: an introduction and catalog — 2023 Edition (blog)](https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/), and now its associated pre-print [Transformer models: an introduction and catalog](https://arxiv.org/abs/2302.07730), "a short and simple catalog and classification of the most popular Transformer models."
    * It includes introductory explanations about transformers and goes over all transformer based models which are either open-source or for which sufficient information has been released to the public. 
    * This is a fairly exhaustive list, and the authors have been updating it every few months.
  * [What LLM to use?](https://github.com/continuedev/what-llm-to-use) is a brief overview of both open-source and commercial **coding-specific** models, up to date October 2023
* To go further: [Complete Transformers Course](https://huggingface.co/learn/nlp-course/chapter1/1) and its [Github repo](https://github.com/huggingface/course)

Why pre-trained?
* See white paper on [Current Best Practices for Training LLMs from Scratch](https://files.catbox.moe/6x8ct9.pdf) by Rebecca Li, Andrea Parker, Justin Tenuto

Important criteria for choosing a model include:

* Model size (in billions of parameters)
  * This is the usual trade-off between computing resources and model abilities. See fine-tuning section and inference sections below for rules of thumb about resources required for various model sizes.
* Context length
  * This may matter for whether the task you hope to accomplish is achievable; and it matters to some extent for the resources required for fine-tuning and inference (see below).
  * If context length matters for the nature of your task, you need to pay close attention to the context length on which the foundation model for the model you are looking at (e.g. LlaMA for all models derived from LlaMA) was trained. 
  * Although nothing prevents anyone from technically increasing the context length of a foundation model, you may experience degeneracy in the output, because while the model becomes technically able to look past e.g. 2048 tokens, it is undertrained to process state vectors of that length. 
  * If you use an already fine-tuned model, pay attention to the length of the data that was used in fine-tuning. If all inputs and outputs were much shorter than the context window, extensive fine-tuning may have degraded the ability of your model to handle long sequences.
  * There maybe an upper limit to the context length that a model of a given size (in billions of parameters) can handle well, given the complexity of long-range pattern.
  * See [this Reddit answer](https://www.reddit.com/r/LocalLLaMA/comments/13ed7re/comment/jjq7mg5/?utm_source=share&utm_medium=web2x&context=3) by an [ExLlama](https://github.com/turboderp/exllama) developper for more on this issue.
* Base model vs instruct/chat version
* License
* Knowledge of training corpus
* Fit between model and task

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

When do I need to fine-tune?

* When using state-of-the-art commercial models (for instance, GPT-4 or Anthropic's Claude), a lot of tasks can be achieved either zero-shot or using in-context learning (providing a few examples in the prompt), with carefully crafted prompts. 
  * In addition, some commercial models may only be available for inference through an API, without access to the weights, in which case it may be impossible to fine-tune it locally.
* Open-source models, on average at this moment, do not perform as well on as large a variety of tasks "out of the box." It is often advantageous to fine-tune them for the specific domain or task you are considering.
  * Open-source models are open, so it is generally possible to access the pre-trained weights for purposes of further training.
* Ultimately, you need to try it and decide -- for my specific use case, is the performance of the pre-trained model "good enough" based on a pre-determined definition of what good means for this task?

## Practical guides

Useful resources:
* [The Novice's LLM Training Guide (blog)](https://rentry.co/llm-training) by [Alpin Dale](https://github.com/AlpinDale)
  * A step-by-step guide to fine-tuning, LoRA, QLoRA, and more.
* [Transformer Math 101 (blog)](https://blog.eleuther.ai/transformer-math/) by Quentin Anthony, Stella Biderman, Hailey Schoelkopf
  * Useful rules of thumb to estimate computation and memory usage for various transformers tasks
  * See also [Model training anatomy (HF docs)](https://huggingface.co/docs/transformers/model_memory_anatomy) for a more detailed explanation of memory use during training
* [Llama 2 Fine-tuning / Inference Recipes and Examples (Github)](https://github.com/facebookresearch/llama-recipes/) and [Hugging Face's Transformers Examples (Github)](https://github.com/huggingface/transformers/tree/main/examples)
  * Avoid reinventing the wheel by first looking for similar examples in the many scripts shared in those two repos
* [Tips for Working with HF on Princeton's Research Computing Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face)
  * Avoid memory and storage issues on the HPC cluster
* [Non-engineers guide: Train a LLaMA 2 chatbot (HF blog)](https://huggingface.co/blog/Llama2-for-non-engineers) by Andrew Jardine and Abhishek Thakur
  * How to fine-tune a LlaMa model for chat without writing any code (this is more of an "at-home" version for those without access to high-performance computing and/or without coding experience)

Datasets for fine-tuning:

* Instruction-following examples: [`databricks-dolly-15k` (dataset)](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
* [Llama Factory (Github)](https://github.com/hiyouga/LLaMA-Factory): code, json datasets, and other resources to help fine-tune some LLMs; supports Llama and Llama-2

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

To go further:

* Two review papers about the training process for transformers:
  * [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) (09-2020)
  * [A Survey on Efficient Training of Transformers](https://arxiv.org/abs/2302.01107) (05-2023)

## Researchers choices in fine-tuning paradigms and methods


* Regularization: 
  * [NEFTUNE: Noisy Embeddings Improve Finetuning](https://arxiv.org/pdf/2310.05914.pdf) (Jain et al. 2023) (Pre-print)

# Aligment

It's important to be aware that fine-tuning a "censored" foundation model can degrade its safety or helpfulness and increase its likelihood to output dangerous or harmful text:

* Foundation models are often created by following three steps: pretraining, fine-tuning, alignment (see the [InstructGPT paper]((https://arxiv.org/abs/2203.02155))). By going back to fine-tune more, you can undo some of the alignment step.
* See [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://arxiv.org/abs/2310.03693) (10-2023)

If your model will be deployed in production, you should consider "re-aligning" it.

[...section in progress...]

* Reinforcement learning from human feedback:
  * The InstructGPT paper: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (03-2022)
  * [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (04-2022)
  * RLHF at home: [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft) (HF blog)
* Alternatives to RLHF:
  * Hindsight Instruction Labeling: [The Wisdom of Hindsight Makes Language Models Better Instruction Followers](https://arxiv.org/abs/2302.05206) (02-2023)
  * Direct Preference Optimization: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (05-2023)
  * Reinforcement Learning with AI Feedback: [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) (09-2023)

# Inference

Here, you have to decide if you will run the model locally on your PC or if you will host it somewhere for inference. Part of this equation is how much computing power and RAM you have available, and how many tokens per second you need to be able to generate.

Note the trade-off between context length and resources required for inference. 

* Because the model, when generating each new token, looks at *all* tokens that came before it, the increase in resources required is more than linear in the context length. That said, although you will often read it is quadratic, in practice, there are many tricks you can use to improve efficiency so that it may end up being closer to linear, especially for quantized models.

Useful resources:

* [Llama 2 Fine-tuning / Inference Recipes and Examples (Github)](https://github.com/facebookresearch/llama-recipes/)
* [Tips for Working with HF on Princeton's Research Computing Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face)

What is a quantized model?

* [To be written]

To go further:

* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (blog)

Some user friendly / laptop friendly tools:

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