# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any

from components.configs import TokenizerConfig
from components.configurable import Configurable
from components.utils import logger
from tokenizers import AddedToken, Tokenizer


class HuggingFaceTokenizer(Configurable):
    Config = TokenizerConfig
    """Loads an HF tokenizer and infers BOS/EOS handling from
    ``tokenizer_config.json``.

    Args:
        config (Config): Configurable config (currently empty).
        tokenizer_path (str): Path to directory containing tokenizer files
            (tokenizer.json, optionally tokenizer_config.json).
    """

    eos_id: int | None

    def __init__(
        self,
        config: TokenizerConfig | None = None,
        *,
        tokenizer_path: str,
    ):
        self.bos_id = None
        self.eos_id = None
        self.bos_token = None
        self.eos_token = None

        self.tokenizer = self._load_tokenizer_from_path(tokenizer_path)

        self._hf_config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )
        if self._hf_config is None:
            logger.warning(
                "No tokenizer_config.json found at %s. "
                "Special token inference disabled.",
                tokenizer_path,
            )

        if self._hf_config is not None:
            self._infer_special_tokens()
        self._infer_should_add_bos_eos()

    def _load_config(self, config_path: str) -> dict | None:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    def _load_tokenizer_from_path(self, tokenizer_path: str) -> Tokenizer:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' does not exist")

        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(
                f"No tokenizer.json found in '{tokenizer_path}'. "
                "See the 'Downloading a tokenizer' section of README.md."
            )
        logger.info("Loading tokenizer from tokenizer.json")
        return Tokenizer.from_file(tokenizer_json_path)

    def _get_token_from_config(self, config: dict[str, Any], key: str) -> str | None:
        """HF tokens are stored as either ``{'bos_token': '<bos>'}`` or
        ``{'bos_token': {'content': '<bos>', ...}}``."""
        token = config.get(key)
        if isinstance(token, dict):
            if "content" not in token:
                raise ValueError(f"Could not parse {key} from config")
            token = token["content"]
        elif token is not None and not isinstance(token, str):
            raise ValueError(
                f"Could not parse {key} from config - expected string or dict"
            )
        return token

    def _process_special_token(
        self, token_str: str, token_config: dict, token_id: int | None = None
    ) -> AddedToken:
        config_bos_token = (
            self._get_token_from_config(self._hf_config, "bos_token")
            if self._hf_config
            else None
        )
        config_eos_token = (
            self._get_token_from_config(self._hf_config, "eos_token")
            if self._hf_config
            else None
        )

        if token_str == config_bos_token:
            self.bos_token = token_str
            self.bos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )
        elif token_str == config_eos_token:
            self.eos_token = token_str
            self.eos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )

        if isinstance(token_config, dict) and (
            token_config.get("__type") == "AddedToken" or "content" in token_config
        ):
            return AddedToken(
                content=token_str,
                single_word=token_config.get("single_word", False),
                lstrip=token_config.get("lstrip", False),
                rstrip=token_config.get("rstrip", False),
                normalized=token_config.get("normalized", True),
                special=token_config.get("special", True),
            )

        return AddedToken(content=token_str, special=True)

    def _infer_special_tokens(self):
        """Read special tokens from ``tokenizer_config.json`` and add them to
        the underlying tokenizer. Stores BOS/EOS as attributes."""
        standard_keys = [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]

        added_tokens_to_add = []

        if not self._hf_config:
            return

        for key in standard_keys:
            token_config = self._hf_config.get(key)
            if token_config is not None:
                token_str = self._get_token_from_config(self._hf_config, key)
                if token_str is not None:
                    added_tokens_to_add.append(
                        self._process_special_token(token_str, token_config)
                    )

        added_tokens_decoder = self._hf_config.get("added_tokens_decoder", {})
        for token_id_str, token_config in added_tokens_decoder.items():
            if isinstance(token_config, dict) and "content" in token_config:
                token_str = token_config["content"]
                added_tokens_to_add.append(
                    self._process_special_token(
                        token_str, token_config, int(token_id_str)
                    )
                )

        if added_tokens_to_add:
            self.tokenizer.add_special_tokens(added_tokens_to_add)
            if self.bos_token:
                self.bos_id = self.tokenizer.token_to_id(self.bos_token)
            if self.eos_token:
                self.eos_id = self.tokenizer.token_to_id(self.eos_token)

    def _infer_should_add_bos_eos(self):
        """Determine whether to add BOS/EOS by config + empirical detection."""
        self.default_add_bos = False
        self.default_add_eos = False
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        encoded_empty_str = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

        if self._hf_config:
            config_add_bos = self._hf_config.get("add_bos_token")
            config_add_eos = self._hf_config.get("add_eos_token")
            if config_add_bos is not None:
                self.default_add_bos = bool(config_add_bos)
            if config_add_eos is not None:
                self.default_add_eos = bool(config_add_eos)

    def encode(self, *args, **kwargs) -> list[int]:
        """Encode text into token IDs with BOS/EOS handling.

        Args:
            text (str): The text to encode.
            add_bos (bool): Whether to add BOS (if not already added by tokenizer).
            add_eos (bool): Whether to add EOS (if not already added by tokenizer).
        """
        text = args[0] if len(args) >= 1 else kwargs.get("text", "")
        add_bos = kwargs.get("add_bos", self.default_add_bos)
        add_eos = kwargs.get("add_eos", self.default_add_eos)

        token_ids = self.tokenizer.encode(text).ids

        if not self.hf_adds_bos and add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)
        if not self.hf_adds_eos and add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids
