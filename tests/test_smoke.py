import asyncio

# A very small smoke test that verifies the package imports and the CLI entrypoint exists.
# Expand with more tests (e.g., mocking Search/Scrape plugins, JSON decision contract, etc.).
def test_imports():
    import convo_orchestrator.main as m
    assert hasattr(m, "main")
