# Chain prompts into a pipeline

Use the output of one model call as the input to the next. This is how you build
applications that do more than answer a single question.

## When to chain

A single model call works when the task is simple: classify a message, extract
data, answer a question. But real tasks often need several steps:

- Extract structured data from a message, then look up the item in a catalog.
- Summarize a document, then translate the summary.
- Classify a message, then route it to the right department, then draft a reply.

Each step is a separate model call. The output of one becomes the input to the
next. This is called prompt chaining or a pipeline.

## Example: extract, look up, reply

Here is a three-step pipeline for the bike shop. A customer asks about a part.
The pipeline extracts the part name, looks up its price, and writes a reply.

```python
from llama_course.conversation import message_text
from llama_course.llm import get_llm
from llama_course.structured import chat_json

llm = get_llm()

# Step 1: extract the part name from the customer message
customer_message = "Do you have the Ridgeline 29 in stock? How much is it?"
extract_schema = {
    "type": "object",
    "properties": {"part_name": {"type": "string"}},
    "required": ["part_name"],
}
messages = [
    {"role": "system", "content": "Extract the product name from the customer message."},
    {"role": "user", "content": customer_message},
]
extracted = chat_json(llm, messages, schema=extract_schema, max_tokens=50, temperature=0.0)
part_name = extracted["part_name"]
print(f"Extracted: {part_name}")

# Step 2: look up the price (in a real app, this would be a database query)
PRICES = {"Ridgeline 29": 1249, "Cityline helmet": 89, "ProLock 1200": 45}
price = PRICES.get(part_name, "unknown")
print(f"Price: {price}")

# Step 3: write a reply using the extracted data
messages = [
    {"role": "system", "content": "You are a bike shop assistant. Write a short, friendly reply."},
    {"role": "user", "content": f"Customer asked: {customer_message}\n\nProduct: {part_name}\nPrice: ${price}\n\nWrite a reply."},
]
reply = message_text(llm.create_chat_completion(messages=messages, max_tokens=100, temperature=0.7))
print(f"\nReply: {reply}")
```

On the 3B model this extracted `Ridgeline 29`, found the price $1249, and replied: "We currently
have the Ridgeline 29 in stock. The price for it is $1249." The price is right. "In stock" is
invented: no step checked stock. Each step only knows what you pass it, and the writing step fills
gaps. Pass every fact the reply needs, or tell it to mention only the facts given.

## When to use tools instead

If step 2 is a database query, an API call, or arithmetic, use tool calling
(lesson 6) instead of chaining. Tools let the model decide when to call your
code, and the code runs once with the exact arguments the model chose. Chaining
is better when every step needs the model's judgment.

## Error handling in pipelines

Each step can fail. The model might return invalid JSON, extract the wrong part
name, or write a reply that doesn't make sense. Handle errors at each step:

```python
from llama_course.structured import InvalidJSONReply

try:
    extracted = chat_json(llm, messages, schema=extract_schema, max_tokens=50, temperature=0.0)
except InvalidJSONReply:
    # Retry once, or fall back to a default
    extracted = {"part_name": "unknown"}
```

For critical pipelines, validate the output of each step before passing it to
the next. If step 1 extracts a part name that doesn't exist in your catalog,
stop and ask the customer to clarify, rather than passing "unknown" to step 3.

## Throughput

Each step is a separate model call, so a three-step pipeline takes three times
as long as a single call. For batch processing (see
[lesson 3](../../lessons/03_project/)), this adds up. Consider:

- Caching intermediate results if the same extraction runs repeatedly.
- Using a smaller, faster model for simple steps (extraction) and a larger
  model for complex steps (reply generation).
- Running multiple pipelines in parallel with separate model instances.

## Where next

- [Lesson 6: Tools](../../lessons/06_tools/): let the model call your code
  instead of chaining manual steps.
- [Cache responses](cache-responses.md): avoid repeating the same model calls.
- [Handle errors](handle-errors.md): what to do when a model call fails.