import pytest

from llama_course.context import (
    MESSAGE_OVERHEAD,
    ContextTooLong,
    count_tokens,
    messages_tokens,
    trim_history,
)


def msg(role, n_words):
    return {"role": role, "content": " ".join(["w"] * n_words)}


def test_count_tokens(fake_llm):
    assert count_tokens(fake_llm([]), "one two three") == 3


def test_messages_tokens_includes_overhead(fake_llm):
    assert messages_tokens(fake_llm([]), [msg("user", 2)]) == 2 + MESSAGE_OVERHEAD


def test_history_that_fits_is_unchanged(fake_llm):
    chat = [msg("system", 1), msg("user", 1), msg("assistant", 1), msg("user", 1)]
    assert trim_history(fake_llm([]), chat, max_tokens=1000) == chat


def test_overflow_drops_oldest_turns_and_keeps_system_and_newest(fake_llm):
    chat = [msg("system", 1), msg("user", 5), msg("assistant", 5), msg("user", 5), msg("assistant", 5), msg("user", 2)]
    budget = (1 + MESSAGE_OVERHEAD) + (5 + MESSAGE_OVERHEAD) * 2 + (2 + MESSAGE_OVERHEAD)
    out = trim_history(fake_llm([]), chat, max_tokens=budget)
    assert out == [chat[0], chat[3], chat[4], chat[5]]  # first turn dropped
    assert messages_tokens(fake_llm([]), out) <= budget


def test_trimmed_history_never_starts_with_an_assistant_reply(fake_llm):
    chat = [msg("user", 5), msg("assistant", 5), msg("user", 1)]
    # Budget fits assistant + newest but not the oldest user message.
    budget = (5 + MESSAGE_OVERHEAD) + (1 + MESSAGE_OVERHEAD)
    out = trim_history(fake_llm([]), chat, max_tokens=budget)
    assert out == [chat[2]]


def test_system_prompt_is_kept_even_when_everything_else_goes(fake_llm):
    chat = [msg("system", 3), msg("user", 50), msg("assistant", 50), msg("user", 2)]
    out = trim_history(fake_llm([]), chat, max_tokens=3 + 2 + 2 * MESSAGE_OVERHEAD)
    assert out == [chat[0], chat[3]]


def test_input_list_is_not_modified(fake_llm):
    chat = [msg("user", 50), msg("assistant", 50), msg("user", 1)]
    trim_history(fake_llm([]), chat, max_tokens=20)
    assert len(chat) == 3


def test_newest_message_too_big_raises(fake_llm):
    with pytest.raises(ContextTooLong):
        trim_history(fake_llm([]), [msg("user", 100)], max_tokens=10)
