

def test_create_cache_in_memory_without_redis(monkeypatch):
    from app.services.llm import client as llm_client

    monkeypatch.setattr(llm_client.settings, "LLM_CACHE_URL", None)
    cache = llm_client._create_cache("test-scope")

    assert isinstance(cache, llm_client._AsyncLRUCache)


def test_create_cache_falls_back_when_redis_initialisation_fails(monkeypatch):
    from app.services.llm import client as llm_client

    monkeypatch.setattr(llm_client.settings, "LLM_CACHE_URL", "redis://example")

    class Boom(Exception):
        pass

    def raising_cache(*args, **kwargs):
        raise Boom("boom")

    monkeypatch.setattr(llm_client, "_RedisCache", raising_cache)

    cache = llm_client._create_cache("test-scope")

    assert isinstance(cache, llm_client._AsyncLRUCache)


def test_create_cache_uses_redis_when_available(monkeypatch):
    from app.services.llm import client as llm_client

    sentinel = object()

    monkeypatch.setattr(llm_client.settings, "LLM_CACHE_URL", "redis://example")
    monkeypatch.setattr(llm_client, "_RedisCache", lambda *args, **kwargs: sentinel)

    cache = llm_client._create_cache("test-scope")
    assert cache is sentinel
