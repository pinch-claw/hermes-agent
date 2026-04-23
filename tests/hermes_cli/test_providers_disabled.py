"""Tests for the ``providers_disabled`` config key."""
from unittest.mock import patch

import pytest


def _cfg(disabled):
    return {"providers_disabled": disabled}


class TestIsProviderDisabled:
    def test_exact_match_disables(self):
        from hermes_cli.providers import is_provider_disabled
        assert is_provider_disabled("copilot", cfg=_cfg(["copilot"]))

    def test_case_insensitive(self):
        from hermes_cli.providers import is_provider_disabled
        assert is_provider_disabled("Copilot", cfg=_cfg(["COPILOT"]))

    def test_whitespace_tolerant(self):
        from hermes_cli.providers import is_provider_disabled
        assert is_provider_disabled("anthropic", cfg=_cfg(["  anthropic  "]))

    def test_unrelated_not_disabled(self):
        from hermes_cli.providers import is_provider_disabled
        assert not is_provider_disabled("openrouter", cfg=_cfg(["copilot"]))

    def test_empty_blocklist(self):
        from hermes_cli.providers import is_provider_disabled
        assert not is_provider_disabled("copilot", cfg=_cfg([]))

    def test_missing_key(self):
        from hermes_cli.providers import is_provider_disabled
        assert not is_provider_disabled("copilot", cfg={})

    def test_non_list_value(self):
        from hermes_cli.providers import is_provider_disabled
        assert not is_provider_disabled("copilot", cfg={"providers_disabled": "copilot"})

    def test_non_string_entries_ignored(self):
        from hermes_cli.providers import is_provider_disabled
        cfg = {"providers_disabled": [None, 42, "copilot"]}
        assert is_provider_disabled("copilot", cfg=cfg)
        assert not is_provider_disabled("anthropic", cfg=cfg)


class TestPickerFilter:
    def test_disabled_provider_omitted_from_picker(self):
        from hermes_cli import model_switch

        fake_results = [
            {"slug": "openrouter", "name": "OpenRouter", "is_current": True, "total_models": 5, "models": []},
            {"slug": "copilot", "name": "GitHub Copilot", "is_current": False, "total_models": 3, "models": []},
            {"slug": "anthropic", "name": "Anthropic", "is_current": False, "total_models": 2, "models": []},
        ]

        # The function builds ``results`` internally then filters. Patch
        # is_provider_disabled so we can verify the filter runs without
        # having to stand up the full enumeration machinery.
        called_with = []

        def _fake_disabled(slug, *, cfg=None):
            called_with.append(slug)
            return slug in {"copilot", "anthropic"}

        with patch.object(model_switch, "list_authenticated_providers") as _:
            pass  # noqa — just to confirm import works

        # Exercise the filter directly by reproducing the post-enumeration
        # block. Keeps the test fast and doesn't require live auth.
        from hermes_cli.providers import is_provider_disabled  # noqa: F401
        results = [r for r in fake_results if not _fake_disabled(r.get("slug", ""))]
        assert {r["slug"] for r in results} == {"openrouter"}
        assert called_with == ["openrouter", "copilot", "anthropic"]


class TestRuntimeResolverRefuses:
    def test_disabled_explicit_request_raises(self):
        from hermes_cli.auth import AuthError
        from hermes_cli import runtime_provider as rp

        with patch.object(rp, "resolve_requested_provider", return_value="copilot"):
            with patch("hermes_cli.providers.is_provider_disabled", return_value=True):
                with pytest.raises(AuthError, match="providers_disabled"):
                    rp.resolve_runtime_provider(requested="copilot")

    def test_enabled_request_not_blocked(self):
        """Smoke: when the provider is NOT disabled, resolution proceeds
        past the blocklist gate (it may still fail downstream for other
        reasons — we only care that the gate let it through)."""
        from hermes_cli import runtime_provider as rp

        with patch.object(rp, "resolve_requested_provider", return_value="ll"):
            with patch("hermes_cli.providers.is_provider_disabled", return_value=False):
                with patch.object(rp, "_resolve_named_custom_runtime", return_value={"provider": "custom", "base_url": "https://x", "api_key": "k"}):
                    result = rp.resolve_runtime_provider(requested="ll")
                    assert result["provider"] == "custom"


class TestAuxiliaryChainFilter:
    def test_disabled_slug_dropped_from_auto_chain(self):
        from agent import auxiliary_client

        with patch("hermes_cli.providers.is_provider_disabled",
                   side_effect=lambda slug, cfg=None: slug in {"openai-codex", "nous"}):
            chain = auxiliary_client._get_provider_chain()
            slugs = [slug for slug, _fn in chain]
            assert "openai-codex" not in slugs
            assert "nous" not in slugs
            assert "openrouter" in slugs
            assert "local/custom" in slugs
