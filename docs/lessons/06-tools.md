# Lesson 6: Tools and validated output

You let the model call Python functions, and you turn its JSON into typed objects that are checked
before use.

**Why it matters:** Models guess at facts and arithmetic. Here the model calls your code for exact answers, and your code checks every call first.

```bash
python lessons/06_tools/01_tool_calling.py
python lessons/06_tools/02_validated_output.py
```

## The ideas

A language model guesses text. It cannot look up your stock or do exact arithmetic. Tool calling
splits the job:

```
question -> model picks {"tool": "get_stock", "arguments": {"item": "helmets"}}
         -> your code validates the arguments and runs the function
         -> the model writes the answer from the result
```

**The model never runs code.** It writes JSON. Only functions you registered in a `ToolBox` can
run, and the example tools are pure: no files, no network, no shell.

**The schema forces a valid call.** `ToolBox.schema()` builds a JSON Schema from the tools: one
option per tool, with its name and argument types, plus `no_tool`. The model cannot name a tool
that does not exist.

**Pydantic validates the arguments.** Each tool describes its arguments as a Pydantic class, for
example `order_total: float = Field(ge=0)`. A negative total is rejected before the function runs.
Pydantic is the check you can trust. The schema only guides generation.

**Errors go back to the model.** An unknown tool, bad arguments or broken JSON are sent back with
an explanation. The model gets one more try, then `ToolCallFailed` is raised. A `ToolError` raised
by a tool itself (for example "unknown item") becomes the result, and the model explains it.

**Why a fresh chat for the answer.** The tool-call chat tells the model to reply with JSON only.
Reusing it made the final answer come back as JSON too, so the answer step starts a new chat.

**Native tool calling.** llama-cpp-python can pass `tools=` to the model's template, but it has no
parser for Llama 3's replies, so this course builds the loop itself.

Lesson 6.2 applies the same idea without tools: `chat_model` asks for JSON matching a Pydantic
class, validates it, and retries once with the error shown to the model. On the 32 supplier
delivery notes the quantities matched on 30, and no reply was invalid. Validation catches a wrong
type or a negative count, not a wrong number: that still needs a check against known answers.

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/tools.py` | `Tool`, `ToolBox`, `ask`, `ToolError`, `ToolCallFailed` |
| `src/llama_course/shop_tools.py` | Three safe example tools |
| `src/llama_course/structured.py` | `chat_model`, `describe_errors`, `InvalidModelReply` |

## Common mistakes

- **Assuming the model will use the tool.** It decides from the tool's description. Vague
  descriptions get ignored.
- **Nested argument models.** The example tools use flat arguments only. Nested Pydantic models
  add `$ref` entries to the schema, which this course did not test. Keep arguments flat.
- **Tools with side effects.** Do not let a model-chosen call delete, send or pay without your
  checks. Validate, and ask a person for anything irreversible.
- **Trusting a 3B model's choices.** It sometimes picks the wrong tool. Show the call, as the
  lesson does, and test the failure path.

## Exercise

Add an `opening_hours` tool that takes a `day` and returns the store hours, then ask "Is the store
open on Sunday?" Compare a short and a long tool description.

<details>
<summary>Answer</summary>

```python
from pydantic import BaseModel, Field
from llama_course.shop_tools import SHOP_TOOLS
from llama_course.tools import Tool, ToolBox, ask

class HoursArgs(BaseModel):
    day: str = Field(description="Weekday name, for example 'sunday'")

HOURS = {"monday": "10 AM to 7 PM", "saturday": "9 AM to 6 PM", "sunday": "closed"}
hours = lambda a: HOURS.get(a.day.lower(), "unknown day")

description = ("Look up the opening hours of our Portland store for a weekday. "
               "Use it for any question about when the store is open or closed.")
box = ToolBox([*SHOP_TOOLS.tools.values(), Tool("opening_hours", description, HoursArgs, hours)])
print(ask(get_llm(), box, "Is the store open on Sunday?", temperature=0.0))
```

With the short description "Opening hours of the Portland store on a given weekday." the 3B model
chose no tool for both "Is the store open on Sunday?" and "What are the store hours on Saturday?" and
guessed. With the long description it called `opening_hours` for the Sunday question and answered
"closed", but it still skipped the tool for the Saturday question. The tool's description decides
whether it is used, and a small model is not reliable at it.
</details>

## Go deeper

- [Pydantic documentation](https://docs.pydantic.dev/latest/): validation, constraints and JSON
  Schema export.
- [Tool use in chat templates](https://huggingface.co/docs/transformers/chat_extras): how models
  are given tool descriptions.
- [Toolformer](https://arxiv.org/abs/2302.04761): models that learn when to call tools.
- [ReAct](https://arxiv.org/abs/2210.03629): interleaving reasoning with tool calls, the idea behind
  most agents.
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/): the format that
  constrains the tool call.
