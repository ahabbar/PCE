from __future__ import annotations

import logging
import os
import re
from typing import Optional

import yaml

logger = logging.getLogger("pce.whitelist")


class ProtocolWhitelist:
    def __init__(self, yaml_path: str = "protocols/whitelist.yaml"):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Whitelist YAML not found: {yaml_path}. "
                "Set PCE_WHITELIST_PATH env var or ensure file exists."
            )
        with open(yaml_path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
        self._data = data
        self._protocols: dict = data["protocols"]
        self._master_tests = self._build_master_test_set()
        logger.info("ProtocolWhitelist loaded %d protocols, %d master tests",
                    len(self._protocols), len(self._master_tests))

    def _build_master_test_set(self) -> set[str]:
        """Flatten master_approved_tests into a set for O(1) lookup."""
        master = self._data.get("master_approved_tests", {})
        return {t.lower() for cat in master.values()
                if isinstance(cat, list) for t in cat}

    def is_allowed(self, protocol_key: str, test_name: str) -> bool:
        """Check test against master list (global guard) then requires_doctor."""
        test_lower = test_name.lower()
        # Must appear in master list first
        in_master = any(test_lower in m or m in test_lower
                        for m in self._master_tests)
        if not in_master:
            return False
        # Must not be in requires_doctor for this protocol
        requires_doc = self.get_requires_doctor(protocol_key)
        return not any(test_lower in r.lower() for r in requires_doc)

    def match(self, chief_complaint: str) -> Optional[str]:
        text = chief_complaint.lower()
        for key, proto in self._protocols.items():
            for kw in proto.get("keywords", []):
                kw_lower = kw.lower()
                # Use word boundary for short ASCII keywords to avoid substring collisions
                if kw_lower.isascii() and len(kw_lower) <= 4:
                    pattern = r"\b" + re.escape(kw_lower) + r"\b"
                    if re.search(pattern, text):
                        return key
                elif kw_lower in text:
                    return key
        return None

    def get_protocol(self, key: str) -> dict:
        if key not in self._protocols:
            raise KeyError(f"Protocol '{key}' not found in whitelist")
        return self._protocols[key]

    def get_auto_orders(self, key: str) -> list[dict]:
        return self._protocols.get(key, {}).get("auto_order", [])

    def get_conditional_orders(self, key: str) -> list[dict]:
        return self._protocols.get(key, {}).get("conditional", [])

    def get_requires_doctor(self, key: str) -> list[str]:
        return self._protocols.get(key, {}).get("requires_doctor", [])

    def all_protocols(self) -> list[str]:
        return list(self._protocols.keys())

    @classmethod
    def from_env(cls) -> ProtocolWhitelist:
        path = os.getenv("PCE_WHITELIST_PATH", "protocols/whitelist.yaml")
        return cls(path)
