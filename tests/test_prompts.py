from llama_course.prompts import few_shot_prompt, llama3_chat_prompt, qa_prompt


def test_qa_prompt_ends_with_answer_cue():
    p = qa_prompt("Explain.", "What is gravity?")
    assert p == "Instruction: Explain.\nQuestion: What is gravity?\nAnswer:\n"


def test_few_shot_numbers_examples_and_leaves_last_label_open():
    p = few_shot_prompt([("Great!", "Positive"), ("Bad.", "Negative")], "Fine.")
    assert p.splitlines() == [
        "Review 1: Great!",
        "Sentiment 1: Positive,",
        "Review 2: Bad.",
        "Sentiment 2: Negative,",
        "Review 3: Fine.",
        "Sentiment 3:",
    ]


def test_few_shot_with_no_examples():
    assert few_shot_prompt([], "Fine.") == "Review 1: Fine.\nSentiment 1:"


def test_llama3_prompt_has_roles_in_order():
    p = llama3_chat_prompt("sys", "usr")
    assert p.startswith("<|start_header_id|>system")
    assert "<|begin_of_text|>" not in p
    assert p.index("system") < p.index("sys<|eot_id|>") < p.index("usr<|eot_id|>")
    assert p.endswith("assistant<|end_header_id|>\n")
