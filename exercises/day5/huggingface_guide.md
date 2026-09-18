# 🤗 Introduction to Hugging Face

---

## 1. 🌐 What the Hub is

Hugging Face is an open platform where researchers and developers share machine learning models, datasets, and demos. The core of the platform is the **Hub**, a registry at [huggingface.co](https://huggingface.co) that you can think of as a kind of GitHub for machine learning.

Anyone can upload a model to the Hub, and as such, models are not very curated. This can make browsing a bit overwhelming. The goal of this guide is to help you navigate the Hub effectively, so that you can find, evaluate, and use models for your own research.

You have already been using the Hub all week, even if you did not visit the website. The Python libraries we worked with (`transformers`, `sentence-transformers`, and `datasets`) download directly from the Hub behind the scenes. When you wrote `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`, for example, the library fetched that model from its Hub page. The first time you load a model it is downloaded and cached locally (under `~/.cache/huggingface/hub/`), so subsequent runs re-use the same download. Most model pages on the Hub also include a ready-made code snippet showing you how to load them.

---

## 2. 🔍 Finding models

The main entry point for browsing models is [huggingface.co/models](https://huggingface.co/models). You will see a long list of models, sorted by trending by default. The key to making this manageable is a combination of the search box and the filters on the left side of the page.

**The search box** matches model names and tags. Searching for an architecture name like `modernbert` or `deberta` finds models built on that base family. Searching for a task word like `sentiment` finds models that have that word in their name.

**The filters** let you narrow results along several dimensions. The most important ones are:

- 📋 **Tasks.** Filters by what the model is trained/fine-tuned for. Generally speaking, the relevant tasks are `Text Classification` for models with a classification head like the offensive speech classifier we fine-tuned, `Sentence Similarity` for embedding models like MiniLM and e5 (that we used for topic modelling), and `Fill-Mask` for base encoder models that have not yet been fine-tuned for a specific task (i.e., are ready for you to fine-tune).
- 🌍 **Languages.** Filters by the languages listed on the model card. Note that if your corpus is single-language, it is generally better to use a model that was base-trained on that language, rather than a multi-lingual model (or at least, to compare the two options in your pipeline).
- 📊 **Sort order.** The default is "trending", which favours recently popular models, which is heavily influenced by industry, rather than research. Sorting by "most downloads" gives you the most widely used models, which is often a reasonable starting point. Download count can also be misleading, though. A widely downloaded model is not necessarily the best one for your task; it may simply be older and/or better known.

A practical approach is to start by filtering for your task and language, sort by downloads to see what is commonly used, and then read the model cards of the top results to assess whether they actually fit your needs.

Many models on the Hub are derivatives of the same base model. You will often see dozens of fine-tuned versions of, say, `google-bert/bert-base-uncased`, each trained on a different dataset for a different task. The base model families covered in Section 4 are the shared starting points from which all of these variants descend. Once you are familiar with the major families, a lot of the apparent variety becomes easier to make sense of.

> ⚠️ Some models on the Hub are **"gated"**, which means you need a free Hugging Face account and must accept the model's terms before you can download them. Some gated models also require a manual approval step. If you get an authentication error when trying to load a model, this is likely the reason.

---

## 3. 📄 Reading a model card

Every model page on the Hub has a README, called the **model card**. It is meant to document what the model is, what it was trained on, and how it should be used. In practice, the quality of model cards varies enormously, especially for 'downstream' models (e.g., finetunes). If you upload your model (e.g., for reproducability), you can do better!

Before you read the card itself, look at the **tags** displayed at the top of the model page. These are set in the card's metadata and show the task, language, licence, library, and training datasets at a glance.

To see what a model card looks like, let's start with the classifier you fine-tuned on Day 3: [`rptkiddle/mmBERT-small-tweeteval-offensive`](https://huggingface.co/rptkiddle/mmBERT-small-tweeteval-offensive). It is short, but it covers the key things a card should cover:

- 🎯 **What the model does.** The opening line says it classifies English tweets as offensive or non-offensive. You immediately know the task, the language, and the domain.
- 🧱 **What base it uses.** The card names its base model (`jhu-clsp/mmBERT-small`) and links to it. This tells you the architecture and lets you trace the model back to its base family (Section 4 covers these families in more detail).
- 💻 **How to use the model.** A code snippet shows you how to load the model with `pipeline("text-classification", ...)`. Many cards include a snippet like this, and it is often the fastest way to get started.
- ⚙️ **Training details.** The card lists the hyperparameters used during fine-tuning (epochs, batch size, learning rate, etc.) and names the training dataset. Note, this is also documented within the model files (config.json).
- 📈 **Evaluation results.** A table shows precision, recall, and F1 for each class on the test set, along with a baseline comparison. This lets you judge how well the model actually performs at the task, over a more simple approach.
- ⚠️ **Limitations.** If you acknowledge limitations of a model in your paper, it's good to also communicate these here.

Now compare this to [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), the embedding model from Day 2. The card is structured differently. There is no "model description" section as such, just a one-line header and a paper reference. The usage section is more technical and includes an essential detail that is easy to miss. Every input must be prefixed with either `"query: "` or `"passage: "`. For everything we did this week (clustering, similarity, topic modelling) the `"query: "` prefix is the right one. If you skip the prefix, the model still produces embeddings, but they will be lower quality. Requirements like this are the main reason to read the card before using a model. The card also includes detailed training data tables (showing a two-stage training process with billions of text pairs) and general benchmark results from MTEB.

Beyond the card, two other parts of the model page are useful to know about:

- 📁 The **Files and versions** tab shows the actual files in the model repository, along with a commit history. Model repos on the Hub are git repositories, so this will feel familiar. The size of the `model.safetensors` file tells you the download size, and for classification models, `config.json` contains the `id2label` mapping, which tells you what the output classes mean (if the author gave them readable names; otherwise you will see generic labels like `LABEL_0`, `LABEL_1`).
- 🔗 The **right-hand sidebar** shows the base model a fine-tune was derived from, when the author has recorded it in the metadata. Click through to the base model's page to see the other models that have been fine-tuned from it. This is a quick way to understand where a model sits in the landscape described in Section 4.

---

## 4. 🗂️ The model taxonomy

The Hub hosts a very large number of encoder models, but most of them descend from a small number of base model families.

### Base encoder families

Here are some of the most important foundation (base) models from the last decade:

| Family | Example | Year | Parameters (base) | Key characteristics |
|--------|---------|------|-------------------|---------------------|
| **BERT** | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) | 2018 | 110M | The original. Still widely used, but older training data, inefficient architecture. |
| **RoBERTa** | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | 2019 | 125M | BERT trained on more and better data. [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) (279M) is the multilingual version and remains widely used. |
| **DeBERTa** | [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | 2021 | 184M | Uses a cleverer masked training objective that helps it contrast similar terms. |
| **ModernBERT** | [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | 2024 | 150M | Lots of fancy features: longer context length, more efficient attention mechanism. `jhu-clsp/mmBERT-small`, the model you fine-tuned on Day 3, is a multilingual model built on this architecture. |

When you encounter a model on the Hub, you can usually trace it back to one of these families by reading the model card or checking the base model in the sidebar.

When choosing a base model to fine-tune, the two main considerations are **language** and **size**. If your data is in one language, a model trained specifically on that language will generally outperform a multilingual one. If you have limited GPU memory, smaller models are practical where larger ones are not. For reference, mmBERT-small is 140M parameters and MiniLM is 23M.

### Three paths from a base model

The reason the Hub has so many models is that each base model can be taken in different directions.

> **Path 1: Base model** (no further training)
>
> These are the pre-trained models as released by their original authors. They understand language, but they are not set up for any specific task. On the Hub, they are tagged as `Fill-Mask`. If you want to fine-tune a classifier on your own labelled data (as you did on Day 3), you start from a base model.

> **Path 2: Contrastive training → embedding model** (tagged `Sentence Similarity` on the Hub)
>
> Someone has taken a base encoder and trained it further using contrastive learning, so that it produces useful sentence-level embeddings. Similar sentences end up with similar vectors; dissimilar sentences end up far apart. This is what makes clustering, similarity search, and topic modelling work.

The major embedding model families for text analysis are:

| Family | Example | Languages | Notes |
|--------|---------|-----------|-------|
| **E5** | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | 100+ | Requires a `"query: "` or `"passage: "` prefix on every input. |
| **BGE** | [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | English and multilingual variants | Similar approach to E5. Widely used. |
| **GTE** | [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) | English and multilingual variants | Newer, competitive. Requires `trust_remote_code=True` when loading. |
| **Nomic** | [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | English | Supports Matryoshka representations (see Section 6). Uses task prefixes (`search_query:`, `clustering:`, etc.). |
| **MiniLM** | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | English | Small and fast. A distilled model (trained to imitate a larger one). Used on Day 1. |

> **Path 3: Task-specific fine-tuning → ready-made classifier** (tagged `Text Classification` or similar on the Hub)
>
> Someone has taken a base model, added a task-specific head (e.g. a classification layer), and trained it on labelled data. The result is a model that performs one specific task well and everything else poorly (finetuning is destructive!). The Day 3 model [`rptkiddle/mmBERT-small-tweeteval-offensive`](https://huggingface.co/rptkiddle/mmBERT-small-tweeteval-offensive) is an example.

---

## 5. 🐍 Loading and using a model

There are three main code patterns for loading models from the Hub. You used Patterns 1 and 3 during the course; Pattern 2 is worth knowing as well. This section collects all three in one place as a reference.

### Pattern 1: `SentenceTransformer` for embeddings

This is what you use when you need vector representations of text, for tasks like clustering, similarity, or topic modelling. It is the simplest pattern.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(["First sentence.", "Second sentence."])
```

The model name is the Hub repository ID. The library handles downloading, caching, and tokenization for you. Some models require a prefix on the input text (see Section 4). You can apply this via the `prompt` argument:

```python
embeddings = model.encode(["First sentence."], prompt="query: ")
```

### Pattern 2: `pipeline` for classification

The `pipeline` function from `transformers` is the easiest way to run a task-specific model. You give it a task name and a model, and it handles tokenization, inference, and decoding in one call.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="rptkiddle/mmBERT-small-tweeteval-offensive")
classifier("You are all wonderful people.")
# [{'label': 'non-offensive', 'score': 0.999}]
```

> 💡 The snippets above do not specify a device. By default, `SentenceTransformer` auto-detects your GPU, but `pipeline` and `AutoModel` default to CPU. In the course notebooks we passed `device=` explicitly; for `pipeline` the equivalent is `pipeline(..., device=0)` for the first GPU.

### Pattern 3: `AutoModel` + `AutoTokenizer`

This is the pattern you used for fine-tuning on Day 3. You load the tokenizer and model separately, which provides more control.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("rptkiddle/mmBERT-small-tweeteval-offensive")
model = AutoModelForSequenceClassification.from_pretrained("rptkiddle/mmBERT-small-tweeteval-offensive")

inputs = tokenizer("This is a test.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = logits.argmax(dim=-1).item()
print(model.config.id2label[predicted_class])  # "non-offensive"
```

The `Auto` classes figure out the correct architecture from the model's configuration, so the same code works for BERT, RoBERTa, DeBERTa, ModernBERT, and any other supported model. When fine-tuning, you would replace `AutoModelForSequenceClassification` with whichever model class matches your task (e.g. `AutoModelForTokenClassification` for named entity recognition).

---

## 6. 📖 Key terms on model cards

When browsing model cards, you will encounter a number of recurring terms. This glossary covers the ones most relevant to text analysis with encoder models.

**Base model vs fine-tuned model.** A base model (e.g. `jhu-clsp/mmBERT-small`) has been pre-trained on general text but is not set up for any specific task. A fine-tuned model (e.g. `rptkiddle/mmBERT-small-tweeteval-offensive`) has been further trained on labelled data to perform a particular task. You cannot use a base model for classification without fine-tuning it first.

**Contrastive learning / contrastive training.** A training method where the model learns from pairs of examples. It is taught that some pairs are similar and others are not, and it adjusts its representations to reflect this. All of the embedding models in Section 4 (MiniLM, E5, BGE, GTE, Nomic) were trained this way. This is what gives them the ability to place similar sentences close together in vector space.

**Hard negatives.** In contrastive training, a hard negative is a pair of texts that are superficially similar but actually have different meanings. Training on hard negatives forces the model to make finer distinctions, which is why models trained this way tend to be better at telling apart texts that are similar but not identical.

**Distillation.** A technique where a smaller model (the "student") is trained to reproduce the outputs of a larger model (the "teacher"). The result is a model that is much faster and smaller, with some loss of quality. MiniLM is a distilled model, which is why it is small (23M parameters) but still produces useful embeddings.

**Prefixes / instruction templates.** Some models expect a short prefix at the beginning of every input, such as `"query: "` for E5 or `"clustering: "` for Nomic. The prefix tells the model what kind of task you are performing, which affects the quality of the embeddings it produces. The model card will document which prefixes are supported and when to use them. Omitting a required prefix does not raise an error, but the embeddings will be lower quality.

**Pooling (CLS vs mean).** An encoder model produces one vector per token. To get a single vector for the whole sentence, these token vectors need to be combined. The two common strategies are *CLS pooling* (using only the vector for the special `[CLS]` token at the start) and *mean pooling* (averaging all token vectors). The pooling strategy is set during training and must match at inference time. For models loaded via `sentence-transformers`, this is handled automatically. If you load a model manually, the model card will tell you which strategy to use.

**Matryoshka Representation Learning (MRL).** A training technique that makes a model's embedding vectors useful at shorter lengths, not just their full dimensionality. For example, Nomic Embed produces 768-dimensional embeddings, but you can truncate them to 256 or even 128 dimensions and still get reasonable results. This can save storage and speed up similarity computations when you have a large corpus. If a model card mentions Matryoshka, it means you have this option.

**Cross-encoder vs bi-encoder.** A *bi-encoder* (like all the embedding models in Section 4) encodes each text independently into a vector, then compares vectors. A *cross-encoder* takes two texts as a single input and produces a similarity score directly. Cross-encoders are more accurate but much slower, because every pair of texts requires a separate forward pass. The two are often used in combination, e.g., a bi-encoder to retrieve a shortlist of candidates and a cross-encoder to re-rank them.

**Max sequence length / context length.** Every encoder model has a maximum number of tokens it can process at once. For the BERT family this is typically 512 tokens; ModernBERT supports 8,192. Any input longer than the limit is truncated. This matters if your documents are longer than a few paragraphs. The Day 3 notebook set `max_length=200` because tweets are short, but for longer documents you would want to use the model's full context window, or split the text into chunks (which you can also take the mean of).

**`trust_remote_code`.** Some models on the Hub ship their own Python code that runs when you load the model. Loading such a model requires passing `trust_remote_code=True`. Only do this for models from authors you trust, since the code runs with some access to your system. GTE (Section 4) is one model family that requires this.

**MTEB.** The Massive Text Embedding Benchmark, a standard suite for evaluating embedding models across many tasks (retrieval, classification, clustering, and others). When a model card reports MTEB scores, these are averages across a large number of datasets. They are useful for comparing models in general terms, but a model's MTEB score may not predict how well it performs on your specific corpus and task.

**Tokenizer.** The component that converts raw text into the numerical tokens the model actually processes. Each model family has its own tokenizer and vocabulary, which is why you always load the tokenizer from the same model repository as the model itself. Mismatching a tokenizer and a model produces nonsensical results.
