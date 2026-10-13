# Serve the classifier with FastAPI

Turn the message classifier from lesson 3 into a web API that other programs on your machine can
call. This is a minimal version to learn the shape, not a production server.

## Install FastAPI

FastAPI is not a course dependency, so install it on its own:

```bash
pip install fastapi uvicorn
```

## The endpoint

Save this as `api.py` in the repository root:

```python
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel

from llama_course.classify import classify
from llama_course.llm import get_llm

app = FastAPI()
llm = get_llm()  # load once, at startup
llm_lock = Lock()


class MessageRequest(BaseModel):
    message: str


class ClassificationResponse(BaseModel):
    category: str


@app.post("/classify", response_model=ClassificationResponse)
def classify_message(req: MessageRequest) -> ClassificationResponse:
    with llm_lock:  # one request at a time: a Llama object is not safe to share between threads
        label = classify(llm, req.message)
    return ClassificationResponse(category=label)
```

FastAPI runs plain `def` endpoints in a pool of threads, so two requests can arrive at the same
time. One `Llama` object holds one conversation state and must not be used by two threads at
once. The lock makes requests wait their turn.

Run it:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

`127.0.0.1` means only programs on this machine can reach it. Use `0.0.0.0` only after you add the
protections listed below, because it opens the port to your whole network.

Test it from a second terminal (tested with four requests sent at once, all answered):

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "My chain jumps when I pedal hard. Can you look at it this week?"}'
```

You get:

```json
{"category":"Repair"}
```

## What this does not do

- Authentication: anyone who can reach the port can use it.
- Speed under load: requests run one at a time. The 3B model sorted 3.7 messages per second on
  an M-series Mac ([`batch_classify.py`](../../lessons/03_project/batch_classify.py), test set). More throughput needs more
  processes, each with its own copy of the model in memory.
- Error reporting: if the model's reply is unusable, `classify` returns the fallback label
  `Feedback` and logs a warning. A real service should tell the caller instead.
- Health checks and request logs.

## Where next

- [Handle errors](handle-errors.md): what to do when a model call fails.
- [Lesson 3: Project](../../lessons/03_project/): the classifier this endpoint wraps.
