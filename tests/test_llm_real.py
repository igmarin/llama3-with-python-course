import pytest

from llama_course.llm import get_llm


@pytest.mark.model
def test_real_model_generates_a_token():
    out = get_llm()("Hello", max_tokens=1)
    assert out["choices"][0]["text"] is not None


@pytest.mark.model
def test_same_seed_gives_same_text():
    llm = get_llm()
    kwargs = {"max_tokens": 20, "temperature": 1.0, "seed": 123}
    first = llm("Write one sentence about the sea.", **kwargs)["choices"][0]["text"]
    second = llm("Write one sentence about the sea.", **kwargs)["choices"][0]["text"]
    assert first == second


@pytest.mark.model
def test_token_count_matches_the_tokenizer():
    from llama_course.context import count_tokens

    llm = get_llm()
    assert count_tokens(llm, "Hello world") == len(llm.tokenize(b"Hello world", add_bos=False)) > 0


@pytest.mark.model
def test_real_embeddings_find_the_returns_document():
    from llama_course.data import DATA_DIR
    from llama_course.llm import get_profile
    from llama_course.rag import Index, llama_embedder, load_chunks

    if not get_profile("embedding").path.is_file():
        pytest.skip("embedding model missing: python scripts/download_model.py --profile embedding")
    emb = get_llm("embedding", embedding=True)
    index = Index(
        load_chunks(DATA_DIR / "docs"),
        llama_embedder(emb, "search_document: "),
        llama_embedder(emb, "search_query: "),
    )
    (chunk, _), *_ = index.search("How long do I have to send a bike back?", k=1)
    assert chunk.source == "returns"
