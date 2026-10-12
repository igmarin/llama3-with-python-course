import pytest

from llama_course.benchmark import (
    RunStats,
    format_table,
    measure_stream,
    peak_memory_mb,
    result_dict,
    run_benchmark,
    summarize,
)


def fake_clock(*times):
    it = iter(times)
    return lambda: next(it)


def test_measure_stream_times_first_token_and_total():
    # start=0, first piece at 2s, end at 12s
    stats = measure_stream(iter(["a", "b", "c", "d", "e", "f"]), fake_clock(0, 2, 12))
    assert stats == RunStats(first_token_s=2, total_s=12, tokens=6)
    assert stats.tokens_per_s == pytest.approx(5 / 10)  # 5 tokens after the first, over 10 s


def test_empty_stream_is_an_error():
    with pytest.raises(ValueError, match="no tokens"):
        measure_stream(iter([]), fake_clock(0, 1))


def test_single_token_has_zero_speed_instead_of_dividing_by_zero():
    assert RunStats(1.0, 1.0, 1).tokens_per_s == 0.0


def test_summarize_uses_the_median():
    runs = [RunStats(1, 11, 11), RunStats(1, 11, 11), RunStats(9, 99, 11)]  # one slow outlier
    s = summarize(runs)
    assert s["first_token_s"] == 1 and s["total_s"] == 11 and s["tokens_per_s"] == pytest.approx(1.0)


def test_run_benchmark_does_one_warmup_then_times_every_run(fake_llm):
    llm = fake_llm(["a b c"] * 7)  # 1 warm-up + 2 prompts x 3 runs
    runs = run_benchmark(llm, ["p1", "p2"], runs=3, max_tokens=10)
    assert len(runs) == 6 and len(llm.calls) == 7
    assert llm.calls[0]["messages"] == llm.calls[1]["messages"]  # warm-up repeats the first prompt
    assert all(r.tokens == 3 for r in runs)
    assert all(c["max_tokens"] == 10 and c["temperature"] == 0.0 for c in llm.calls)


def test_peak_memory_units():
    assert peak_memory_mb(2 * 1024**3, "darwin") == pytest.approx(2048)  # bytes on macOS
    assert peak_memory_mb(2 * 1024**2, "linux") == pytest.approx(2048)  # kilobytes on Linux


def test_result_dict_and_table():
    runs = [RunStats(0.5, 5.5, 51)]
    result = result_dict("laptop-16gb", 3.2, runs, 2100.0, {"platform": "Darwin arm64", "ram_gb": 16.0})
    assert result["runs"][0]["tokens_per_s"] == pytest.approx(10.0)
    table = format_table(result)
    assert "laptop-16gb" in table and "10.0 tokens/s" in table and "16 GB" in table
