"""Unit tests - no network, no LLM provider, no Docker.

Everything here runs offline and deterministically. Tests that need a live
provider carry `@pytest.mark.live` and are deselected by default via pytest.ini.
"""
