"""ITEP-98577 / GitHub #153: the ASR priming hint must not hardcode "QuickBite
Express restaurant" when a different menu is loaded.

``_build_asr_prompt`` now reads an optional ``restaurant_name`` key from the
active ``products.yaml`` instead of hardcoding QuickBite's name, and omits the
restaurant-name prefix entirely when the key is absent.
"""
import importlib

import yaml

from kiosk_core import config as kiosk_config


def _build_prompt_for(monkeypatch, tmp_path, data: dict) -> str:
    products_yaml = tmp_path / "products.yaml"
    products_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("KIOSK_CORE_PRODUCTS_YAML", str(products_yaml))
    # _ASR_PROMPT_FALLBACK stays module-level; only exercise the builder, not
    # the module-import-time DEFAULT_ASR_PROMPT constant.
    importlib.reload(kiosk_config)
    return kiosk_config._build_asr_prompt()


class TestBuildAsrPrompt:
    def test_uses_restaurant_name_from_active_menu(self, monkeypatch, tmp_path):
        prompt = _build_prompt_for(
            monkeypatch,
            tmp_path,
            {
                "restaurant_name": "Ranger Burgers",
                "products": [{"name": "Ranger Classic Burger"}],
            },
        )
        assert prompt.startswith("Ranger Burgers restaurant. ")
        assert "QuickBite" not in prompt

    def test_omits_restaurant_prefix_when_key_absent(self, monkeypatch, tmp_path):
        prompt = _build_prompt_for(
            monkeypatch,
            tmp_path,
            {"products": [{"name": "Some Item"}]},
        )
        assert not prompt.startswith("QuickBite")
        assert "restaurant." not in prompt
        assert prompt.startswith("Some Item")
