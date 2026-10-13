# Glossary

Every term the course uses, in plain words. The last column links to where each term is first explained.

| Term | Meaning | Explained in |
|------|---------|--------------|
| Accuracy | The share of answers that are correct | [3](../lessons/03_project/) |
| Adapter | The small add-on that LoRA trains: a file of a few megabytes, loaded on top of the original model | [7](../lessons/07_finetuning/) |
| Challenge set | A second, harder test set of 60 hard messages, used only for scoring | [3](../lessons/03_project/) |
| Chat template | The rules for writing roles into the text the model reads, using special tokens | [1](../lessons/01_basics/) |
| Chunk | A short piece of a document, small enough to put in a prompt | [5](../lessons/05_rag/) |
| Classifier | A program that puts each input into one of a fixed set of categories (labels) | [3](../lessons/03_project/) |
| Completion | The text the model writes back, continuing your prompt | [1](../lessons/01_basics/) |
| Confusion matrix | A table of true category against predicted category, showing which ones get mixed up | [3](../lessons/03_project/) |
| Context cache | Memory the model uses to hold your conversation while it works. A bigger context window needs more | [models](models.md) |
| Context window | The most tokens the model can read at once (`n_ctx`), counting both your chat and its reply | [4](../lessons/04_generation/) |
| Cosine similarity | A score for how close two embeddings point: 1.0 is the same direction, near 0 is unrelated | [5](../lessons/05_rag/) |
| Embedding | A list of numbers (a vector) that represents the meaning of a text | [5](../lessons/05_rag/) |
| Environment variable | A setting the terminal passes to a program, such as `LLAMA_PROFILE=laptop-8b` | [how-to](howto/switch-models.md) |
| Few-shot prompting | Showing the model a few solved examples in the prompt so it copies the pattern | [2](../lessons/02_prompting/) |
| Fine-tuning | Training an existing model a little more, on your own examples | [7](../lessons/07_finetuning/) |
| GGUF | The model file format that llama.cpp reads. One file holds the weights and settings | [1](../lessons/01_basics/) |
| GPU | The chip that does most of the model's arithmetic. On a Mac it shares memory with the rest of the computer | [models](models.md) |
| Grammar | A set of rules that limits which tokens the model may pick next | [2](../lessons/02_prompting/) |
| Hugging Face | The website where most open models, including the course's GGUF files, are published | [how-to](howto/add-a-model-profile.md) |
| JSON | A text format for data, made of keys and values, that Python reads with `json.loads` | [2](../lessons/02_prompting/) |
| JSON Schema | A description of the JSON you expect: which keys, and what type each value has | [2](../lessons/02_prompting/) |
| Language model | A program that has learned, from a lot of text, to guess what text comes next | [1](../lessons/01_basics/) |
| Layer | One of the stacked processing stages inside a model | [how-to](howto/add-a-model-profile.md) |
| LoRA | Low-rank adaptation: fine-tuning that leaves the model unchanged and trains a small add-on instead | [7](../lessons/07_finetuning/) |
| Loss | A number for how wrong the model's answers are on a set of examples. Lower is better | [7](../lessons/07_finetuning/) |
| Precision | Of the messages the model put in a category, the share that really belong there | [3](../lessons/03_project/) |
| Profile | A named model setup in configs/models.yaml, chosen with LLAMA_PROFILE | [models](models.md) |
| Prompt | The text you send to the model | [1](../lessons/01_basics/) |
| Prompt injection | Text that tries to give the model new orders, hidden in content it reads | [5](../lessons/05_rag/) |
| Pydantic | A Python library that checks data against a class you define: types, ranges and required fields | [6](../lessons/06_tools/) |
| Quantization | Storing each weight in fewer bits (for example 4.5 instead of 16) to make the model smaller and faster | [1](../lessons/01_basics/) |
| RAG | Retrieval-augmented generation: find the relevant text first, then let the model answer from it | [5](../lessons/05_rag/) |
| Recall | Of the messages that really belong in a category, the share the model found | [3](../lessons/03_project/) |
| ROUGE | A score for generated text, based on how many words it shares with a human-written version | [7](../lessons/07_finetuning/) |
| Sampling | How the model picks the next token from all the likely ones | [4](../lessons/04_generation/) |
| Seed | A number that fixes the random choices, so a run can be repeated exactly | [4](../lessons/04_generation/) |
| Streaming | Receiving the reply token by token while it is written, instead of all at the end | [4](../lessons/04_generation/) |
| System prompt | The first message in a chat, from you, setting rules for the assistant | [1](../lessons/01_basics/) |
| Temperature | A setting for how random that pick is. 0 means no randomness | [4](../lessons/04_generation/) |
| Test split | Data kept aside and scored once at the end, to see how the classifier does on new inputs | [3](../lessons/03_project/) |
| Token | A small piece of text the model works with: a short word, part of a longer word, or a symbol | [1](../lessons/01_basics/) |
| Tool | A Python function the model is allowed to ask for, such as a stock lookup | [6](../lessons/06_tools/) |
| Tool call | The model's request to run a tool, written as JSON: the tool's name and its arguments | [6](../lessons/06_tools/) |
| Training split | Data you are allowed to look at while you design and adjust the classifier | [3](../lessons/03_project/) |
| Validation | Checking data before you use it, and rejecting it if it is wrong | [6](../lessons/06_tools/) |
| Validation set | Examples the model never trains on, used to check for memorising during training | [7](../lessons/07_finetuning/) |
| Virtual environment | A private copy of Python for one project, so its packages don't affect anything else | [setup](setup.md) |
| Weights | The numbers the model learned during training. A "3B" model has about 3 billion of them | [1](../lessons/01_basics/) |
| Zero-shot | Asking without any examples | [2](../lessons/02_prompting/) |
