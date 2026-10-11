from llama_course.conversation import Conversation, message_text


def test_history_order_across_turns(fake_llm):
    llm = fake_llm(["Madrid", "Barcelona"])
    chat = Conversation(llm, system_prompt="Be brief.")
    assert chat.create_completion("A city?") == "Madrid"
    assert chat.create_completion("Another?") == "Barcelona"
    assert [m["role"] for m in chat.history] == ["system", "user", "assistant", "user", "assistant"]
    # Second call saw the whole first turn.
    assert [m["content"] for m in llm.calls[1]["messages"]] == ["Be brief.", "A city?", "Madrid", "Another?"]


def test_kwargs_reach_the_model(fake_llm):
    llm = fake_llm(["ok"])
    Conversation(llm).create_completion("hi", max_tokens=5)
    assert llm.calls[0]["max_tokens"] == 5


def test_empty_system_prompt_adds_no_system_message(fake_llm):
    chat = Conversation(fake_llm(["ok"]))
    chat.create_completion("hi")
    assert chat.history[0]["role"] == "user"


def test_conversations_do_not_share_history(fake_llm):
    a = Conversation(fake_llm(["x"]), system_prompt="A")
    b = Conversation(fake_llm(["y"]), system_prompt="B")
    a.create_completion("hi")
    assert len(b.history) == 1


def test_initial_history_is_copied(fake_llm):
    start = [{"role": "user", "content": "earlier"}]
    chat = Conversation(fake_llm(["ok"]), history=start)
    chat.create_completion("now")
    assert start == [{"role": "user", "content": "earlier"}]


def test_message_text():
    assert message_text({"choices": [{"message": {"content": "hi"}}]}) == "hi"
