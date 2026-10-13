from llama_course.prompts import few_shot_prompt, llama3_chat_prompt, qa_prompt


def test_qa_prompt_ends_with_answer_cue():
    p = qa_prompt("Explain.", "What do gears do?")
    assert p == "Instruction: Explain.\nQuestion: What do gears do?\nAnswer:\n"


def test_few_shot_numbers_examples_and_leaves_last_label_open():
    p = few_shot_prompt([("Flat path", "Easy"), ("Alpine pass", "Hard")], "Hilly loop", "Route", "Difficulty")
    assert p.splitlines() == [
        "Route 1: Flat path",
        "Difficulty 1: Easy,",
        "Route 2: Alpine pass",
        "Difficulty 2: Hard,",
        "Route 3: Hilly loop",
        "Difficulty 3:",
    ]


def test_few_shot_default_labels_and_no_examples():
    assert few_shot_prompt([], "Hilly loop") == "Input 1: Hilly loop\nLabel 1:"


def test_llama3_prompt_has_roles_in_order():
    p = llama3_chat_prompt("sys", "usr")
    assert p.startswith("<|start_header_id|>system")
    assert "<|begin_of_text|>" not in p
    assert p.index("system") < p.index("sys<|eot_id|>") < p.index("usr<|eot_id|>")
    assert p.endswith("assistant<|end_header_id|>\n")
