# Serve the classifier with FastAPI

Turn the message classifier from lesson 3 into a web API that other programs
can call. This guide shows a minimal FastAPI endpoint. It's not
production-ready, but it shows the shape.

## Install FastAPI

```bash
pip install fastapi uvicorn
```

## The endpoint

Save this as `api.py` in the repository root:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from llama_course.classify import classify
from llama_course.llm import get_llm

app = FastAPI()
llm = get_llm()  # load once at startup

class MessageRequest(BaseModel):
    message: str

class ClassificationResponse(BaseModel):
    category: str

@app.post("/classify", response_model=ClassificationResponse)
def classify_message(req: MessageRequest):
    label = classify(llm, req.message)
    return ClassificationResponse(category=label)
```

Run it:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "My chain jumps when I pedal hard. Can you look at it this week?"}'
```

You get:

```json
{"category": "Repair"}
```

## What this doesn't do

This is a starting point, not a production API. It's missing:

- **Authentication**: anyone who can reach port 8000 can call the endpoint. Add
  API keys or OAuth.
- **Rate limiting**: a client can send thousands of requests per second. Add a
  rate limiter (e.g., `slowapi`).
- **Timeouts**: the model can hang on a long prompt. Add a timeout to the
  endpoint (e.g., `asyncio.wait_for`).
- **Error handling**: if the model fails, the endpoint returns a 500. Catch
  exceptions and return a useful error message.
- **Logging**: log each request and response for debugging.
- **Health checks**: add a `/health` endpoint that returns 200 if the model is
  loaded.
- **Batch endpoint**: accept a list of messages and return a list of labels.

## Batch endpoint

Process multiple messages in one request:

```python
class BatchRequest(BaseModel):
    messages: list[str]

class BatchResponse(BaseModel):
    categories: list[str]

@app.post("/classify/batch", response_model=BatchResponse)
def classify_batch(req: BatchRequest):
    labels = [classify(llm, msg) for msg in req.messages]
    return BatchResponse(categories=labels)
```

This runs the model sequentially. For parallel processing, use
`asyncio` and multiple model instances (one per process).

## Where next

- [Handle errors](handle-errors.md): add timeouts and retries to the endpoint.
- [Cache responses](cache-responses.md): avoid re-classifying the same message.
- [Lesson 3: Project](../../lessons/03_project/): the classifier this endpoint
  wraps.