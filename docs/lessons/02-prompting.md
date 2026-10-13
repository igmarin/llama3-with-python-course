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
`few_shot_prompt` builds the text. Lesson 2.2 rates cycling routes as Easy, Moderate or Hard. Its
new route is harder than every example, and with examples alone the 3B model answers `Very Hard`, a
label that does not exist. Examples teach the format, not the list of allowed answers. One line
naming the labels fixes it. Lesson 3 shows that examples can also help on some inputs and hurt on
others.

**JSON mode.** `response_format={"type": "json_object"}` forces valid JSON. Lesson 2.3 turns a
supplier's delivery note into a dictionary of parts and quantities. llama.cpp does this by
restricting which tokens the model may pick at each step, using a grammar. A JSON Schema goes further
and fixes the keys and types: lesson 2.4 pulls a name, a price and a weight out of a product
description. `chat_json` sends the request and parses the reply.

**Memory is your job.** The model remembers nothing between calls. `Conversation` keeps the list
of messages and sends all of it each turn, which is why "a different city in the same country"
works in lesson 2.5: a bike recommendation, then a follow-up about the same ride. The list grows every turn; lesson 4 covers the limit.

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/prompts.py` | `qa_prompt`, `few_shot_prompt` |
| `src/llama_course/structured.py` | `chat_json`, and `InvalidJSONReply` for a reply that is not JSON |
| `src/llama_course/conversation.py` | `Conversation`: history, one method per turn |

## Common mistakes

- **A reply cut off mid-JSON.** If `max_tokens` is too small, the JSON stops mid-string and
  parsing fails with `InvalidJSONReply`. Leave room for the whole object.
- **Valid JSON, wrong content.** JSON mode guarantees syntax, not truth. The model can still
  miscount. Check the values you care about.
- **Sharing a mutable default.** An earlier version of `Conversation` used `history=[]` as a
  default argument, so every chatbot shared one list. The fix is `history=None`. A test covers it.
- **Too many examples.** Each example is text the model copies. Pick examples that cover the cases
  you care about, and test with and without them.

## Exercise

In lesson 2.4, add a `frame_material` string field to `SCHEMA` and to `required`. The description
never says what the frame is made of. What does the model return? How do you let it say "unknown"?

<details>
<summary>Answer</summary>

It returns `'frame_material': 'hardtail'`. The schema demands a string, so the model must write
something, and it takes the nearest word in the text, which describes the frame's shape, not its
material. A required field forces an answer even when there is none.

Allow `null` with `"frame_material": {"type": ["string", "null"]}`, and tell the model in the system
prompt: "Use null for anything the description does not state." Then the 3B model returns
`'frame_material': None`. (Both runs at temperature 0.)
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
