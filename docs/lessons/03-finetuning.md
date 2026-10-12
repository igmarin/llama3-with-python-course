# Lesson 3: Fine-tuning

You train a small add-on for the model on your own examples, then check whether it helped. You also
learn how to score generated text.

```bash
pip install -e ".[finetune]"     # Apple Silicon only
python lessons/03_finetuning/01_prepare_data.py
python lessons/03_finetuning/02_train_lora.py
python lessons/03_finetuning/03_evaluate_tuned_model.py
python lessons/03_finetuning/04_rouge_summaries.py
```

## The ideas

**Fine-tuning teaches by example.** Instead of describing the task in the prompt, you show the model
many pairs: an email, and the right label. The prompt can then be short.

**LoRA trains a small add-on.** Changing every weight of a 3B model needs far more memory than a
laptop has. LoRA (low-rank adaptation) freezes the model and trains small extra matrices instead.
Here that is 0.216% of the parameters (6.9 million of 3.2 billion). The result is an adapter of a few
megabytes that sits on top of the original model. Training on a 4-bit model is called QLoRA. It is what lets
the 3B model train in 16 GB. The course uses `mlx-lm`, which runs on Apple Silicon.

**Three data sets, three jobs.**

| File | Used for |
|------|----------|
| `train.jsonl` (111) | The model learns from these |
| `valid.jsonl` (18) | Checked during training to spot memorising. The model never learns from it |
| `test.jsonl` (33) | The final exam. Looked at once |

**Validation loss.** Loss measures how wrong the model's answers are. Training loss always falls.
If validation loss starts to rise while training loss keeps falling, the model is memorising the
training emails. Choose the number of training steps from validation loss.

**Accuracy and the confusion matrix.** For labels, accuracy is the share of correct answers. A
confusion matrix shows which labels get mixed up, so you can see that Priority emails are being
called Updates. Lesson 3.3 reports easy and hard emails separately, because a model can score well
on easy emails and fail the hard ones.

**ROUGE.** A summary has no single right answer, so lesson 3.4 compares it with a human summary by
word overlap. ROUGE-1 counts shared words, ROUGE-2 shared word pairs, ROUGE-L the longest shared
sequence. It is crude: a good summary in different words scores low. Use it to compare two models
on the same texts.

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/finetune.py` | Training examples as chats, the validation split, `parse_label` |
| `src/llama_course/classify.py` | `score_predictions`, the metrics shared with lesson 4 |
| `src/llama_course/rouge.py` | `rouge_scores` |
| `configs/models.yaml` | `mlx_repo`: the MLX copy of each model |

## Results on the 3B model

| | Base | Tuned (60 steps) |
|---|------|-------|
| All 33 test emails | 90.9% | 93.9% |
| 6 hard emails | 83.3% | 100% |

The test set is small: one email is 3 points. Treat a difference of one or two emails as noise.
The base model here gets a short prompt with no rules. The rules-and-schema prompt from lesson 4
scores 100% on the same emails, so a good prompt beat fine-tuning on this easy synthetic data.
Fine-tuning helps when a prompt gets long, slow or fragile, or when the task has no short rules.

## Common mistakes

- **Choosing settings from the test score.** If you try several settings and keep the one with the
  best test score, the test score no longer tells you how the model does on new emails. Choose
  from validation loss, then run the test once.
- **Training too long.** The training loss reached 0 while validation loss rose.
- **Judging on the training data.** A model that memorised its examples scores 100% on them.
- **Comparing unequal setups.** Base and tuned must get the same prompt and the same emails.

## Exercise

In `02_train_lora.py`, set `ITERS = 200` and run lessons 3.2 and 3.3 again. What happens to
validation loss, and to the test accuracy? What should you conclude?

<details>
<summary>Answer</summary>

Validation loss was 0.065 at step 50, then 0.200, 0.180 and 0.175 at steps 100, 150 and 200,
while training loss fell to 0.000. The model memorises the training emails. With 60 steps the
validation loss was 0.053, 0.071 and 0.114 at steps 20, 40 and 60.

Test accuracy at 200 steps was 97.0%, higher than the 93.9% at 60 steps. That is a one-email
difference on 33 emails, and it is not a reason to pick 200: choosing steps by test score would
spend the test set. The course picks the setting from validation loss.
</details>

## Go deeper

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): the
  original paper.
- [QLoRA](https://arxiv.org/abs/2305.14314): fine-tuning on a 4-bit model.
- [mlx-lm LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md): every option of
  `mlx_lm lora`.
- [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html): Apple's array
  framework under `mlx-lm`.
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index): the same idea on PyTorch, the usual
  route on Linux and NVIDIA GPUs.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/): the
  paper that defines the scores.
- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html):
  precision, recall and confusion matrices explained.
