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
        self._protocols: dict = data["protocols"]
        logger.info("ProtocolWhitelist loaded %d protocols", len(self._protocols))

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

    def is_allowed(self, protocol_key: str, test_name: str) -> bool:
        if protocol_key not in self._protocols:
            return False
        proto = self._protocols[protocol_key]
        test_lower = test_name.lower()

        requires_doctor = [r.lower() for r in proto.get("requires_doctor", [])]
        for rd in requires_doctor:
            if rd.startswith(test_lower) or test_lower.startswith(rd):
                return False

        all_orders = proto.get("auto_order", []) + proto.get("conditional", [])
        for order in all_orders:
            wl_lower = order["test"].lower()
            if wl_lower.startswith(test_lower) or test_lower.startswith(wl_lower):
                return True
        return False

    def get_requires_doctor(self, key: str) -> list[str]:
        return self._protocols.get(key, {}).get("requires_doctor", [])

    def all_protocols(self) -> list[str]:
        return list(self._protocols.keys())

    @classmethod
    def from_env(cls) -> ProtocolWhitelist:
        path = os.getenv("PCE_WHITELIST_PATH", "protocols/whitelist.yaml")
        return cls(path)
