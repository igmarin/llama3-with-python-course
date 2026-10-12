# Lesson 2: Prompting

You shape what the model gives back: its layout, its examples, and a JSON format your program can
read.

```bash
python lessons/02_prompting/01_prompt_formatting.py
python lessons/02_prompting/02_few_shot.py
python lessons/02_prompting/03_json_inventory.py
python lessons/02_prompting/04_json_schema.py
python lessons/02_prompting/05_conversation_class.py
```

## The ideas

**Format the prompt so the answer comes next.** A completion model continues your text. Lesson 2.1
writes `Instruction:`, `Question:` and `Answer:`, so the likeliest continuation is an answer. A
`stop` string such as `"Question:"` ends generation before the model invents the next question.

**Few-shot prompting.** Show a few input and label pairs and leave the last label blank. The model
copies the pattern. This is called few-shot prompting; giving no examples is zero-shot.
`few_shot_prompt` builds the text. Examples are not free: lesson 4 shows a case where six examples
made the model worse.

**JSON mode.** `response_format={"type": "json_object"}` forces valid JSON. llama.cpp does this by
restricting which tokens the model may pick at each step, using a grammar. A JSON Schema goes further
and fixes the keys and types (lesson 2.4). `chat_json` sends the request and parses the reply.

**Memory is your job.** The model remembers nothing between calls. `Conversation` keeps the list
of messages and sends all of it each turn, which is why "a different city in the same country"
works in lesson 2.5. The list grows every turn; lesson 5 covers the limit.

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/prompts.py` | `qa_prompt`, `few_shot_prompt` |
| `src/llama_course/structured.py` | `chat_json`, and `InvalidJSONReply` for a reply that is not JSON |
| `src/llama_course/conversation.py` | `Conversation`: history, one method per turn |

## Common mistakes

- **A reply cut off mid-JSON.** If `max_tokens` is too small, the JSON stops mid-string and
  parsing fails with `InvalidJSONReply`. Lesson 2.4 asks for a short answer and leaves room.
- **Valid JSON, wrong content.** JSON mode guarantees syntax, not truth. The model can still
  miscount. Check the values you care about.
- **Sharing a mutable default.** An earlier version of `Conversation` used `history=[]` as a
  default argument, so every chatbot shared one list. The fix is `history=None`. A test covers it.
- **Too many examples.** Each example is text the model copies. Pick examples that cover the cases
  you care about, and test with and without them.

## Exercise

Classify "It was fine. Not good, not bad." with the two examples in lesson 2.2. Then add a third
example labelled `Neutral` and run it again. What does the model print each time?

<details>
<summary>Answer</summary>

Both runs print `Neutral,`. Even with only Positive and Negative examples the 3B model chose
`Neutral`, because it already knows the word. What the examples fix is the format: one label
word, followed by a comma. That is why the trailing comma appears in the output. Examples mostly
teach the shape of the answer. They matter more when a label is unusual, or when the model's own
choice would be wrong. Add the neutral example if you want the model to use `Neutral` reliably.
</details>

## Go deeper

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): the paper that
  showed models learn tasks from examples in the prompt.
- [Prompt Engineering Guide](https://www.promptingguide.ai/): a catalogue of prompting techniques.
- [Claude prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview):
  practical advice that applies to most chat models.
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/): the language
  lesson 2.4 uses to fix keys and types.
- [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md): how
  output is restricted to a format.
