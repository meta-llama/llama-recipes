# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os

from abc import ABC, abstractmethod
from logging import getLogger
from typing import (
    Any,
    List,
    Iterator,
    Dict,
    Optional,
    Union,
    Literal,
    AbstractSet,
    Collection
)


# +
try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    
except ImportError:
    raise ImportError(
        "Please install tiktoken, blobfile and, lxml with `pip install tiktoken blobfile lxml`."
)

# +
logger = getLogger()

class BaseTokenizer(ABC):
    def __init__(self, model_path: str) -> None:
        assert os.path.exists(
            model_path
        ), f"The tokenizer path does not exist: {model_path}"
        self._bos_id = 0
        self._eos_id = 1
        self._pad_id = -1
        self._n_words = 2

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def n_words(self) -> int:
        return self._n_words

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> List[int]: ...

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> str: ...


# +
# pyre-strict
import os
from dataclasses import dataclass

@dataclass
class TokenizerArgs():
    """
    A data class that holds the arguments for a tokenizer. Default value is set from llama3 model
    Attributes:
        model (str): The name of the model to be used for tokenization. Default is "cl_toplang_128k".
        directory (str): The directory where the model is located. Default is an empty string.(not exposed yet)
        tokenizer_cls (str): The class name of the tokenizer. Default is "TiktokenTokenizer".
        num_reserved_special_tokens (int): The number of special tokens reserved. Default is 256.
    Property:
        path (str): The full path to the model, combining the directory and model name.
    """

    model: str = "cl_toplang_128k"
    directory: str = ""
    tokenizer_cls: str = "TiktokenTokenizer"
    num_reserved_special_tokens: int = 256

    @property
    def path(self) -> str:
        return os.path.join(self.directory, self.model)


# -

def split_whitespaces_or_nonwhitespaces(
    s: str, max_consecutive_slice_len: int
) -> Iterator[str]:
    """
    Split the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
        is_now_space = s[i].isspace()

        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


class TiktokenTokenizer(BaseTokenizer):
    BASIC_SPECIAL_TOKENS = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ]
    CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __init__(self, args: TokenizerArgs) -> None:
        super().__init__(args.path)

        mergeable_ranks = load_tiktoken_bpe(args.path)
        all_special_tokens_with_ids = self._get_all_special_tokens_with_ids(
            args, len(mergeable_ranks)
        )

        self._tok_model = tiktoken.Encoding(
            name=args.model,
            pat_str=TiktokenTokenizer.CL100K_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens={**all_special_tokens_with_ids},
        )
        logger.info(f"Reloaded Tiktoken model from {args.path}")

        self._bos_id: int = self.encode(
            TiktokenTokenizer.BASIC_SPECIAL_TOKENS[0],
            bos=False,
            eos=False,
            allowed_special="all",
        )[0]
        self._eos_id: int = self.encode(
            TiktokenTokenizer.BASIC_SPECIAL_TOKENS[1],
            bos=False,
            eos=False,
            allowed_special="all",
        )[0]
        self._pad_id = -1
        self._n_words: int = self._tok_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def _get_all_special_tokens_with_ids(
        self, args: TokenizerArgs, num_base_tokens: int
    ) -> Dict[str, int]:

        all_special_tokens = TiktokenTokenizer.BASIC_SPECIAL_TOKENS

        assert len(set(all_special_tokens)) == len(
            all_special_tokens
        ), "Special tokens must be unique."

        n_vocab = num_base_tokens + args.num_reserved_special_tokens
        assert (
            n_vocab % 8 == 0
        ), "Vocabulary size must be divisible by 8 for vocabulary parallelism on 8 GPUs"

        assert (
            len(all_special_tokens) <= args.num_reserved_special_tokens
        ), "The total number of basic and extra special tokens exceeds the number of reserved tokens."

        reserved_tokens = [
            f"<|reserved_special_token_{i}|>"
            for i in range(args.num_reserved_special_tokens - len(all_special_tokens))
        ]
        all_special_tokens = (
            all_special_tokens[:-1] + reserved_tokens + [all_special_tokens[-1]]
        )

        return {
            token: num_base_tokens + i for i, token in enumerate(all_special_tokens)
        }

    def encode(
        self,
        s: str,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
        *args: Any,
        **kwargs: Any,
    ) -> List[int]:
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException (may go beyond 400k)
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # Tiktoken is very bad at handling long sequences where either no whitespaces or only whitespaces:
        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consequtive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        # TODO check if MAX_NO_WHITESPACES_CHARS already fixes the issue with TIKTOKEN_MAX_ENCODE_CHARS

        substrs: List[str] = []
        t = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            substr = s[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
            sliced_substr = split_whitespaces_or_nonwhitespaces(
                substr, MAX_NO_WHITESPACES_CHARS
            )
            substrs.extend(sliced_substr)
        for substr in substrs:
            # By default, setting disallowed_special=() encodes a string by
            # ignoring special tokens. Specifically:
            # - Setting `disallowed_special` to () will cause all text
            #   corresponding to special tokens to be encoded as natural
            #   text (insteading of raising an error).
            # - Setting `allowed_special` to "all" will treat all text
            #   corresponding to special tokens to be encoded as special tokens
            t.extend(
                self._tok_model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(
        self,
        tokens: List[int],
        cut_at_eos: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        tokens = [token for token in tokens if token not in [self.bos_id, self.eos_id]]
        return self._tok_model.decode(tokens)


if __name__ == '__main__' and '__file__' not in globals():
    args = TokenizerArgs()
    args.directory = "/home/ubuntu/projects/llama/models/llama_guard-v2/"
    
    tokenizer = TiktokenTokenizer(args)
    
    print(tokenizer.encode("Hello World!", True, True))



# +
#WIP




class SimpleTiktokenTokenizer:
    """tokenizing and encoding/decoding text using tiktoken."""

    BASIC_SPECIAL_TOKENS = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ]
    CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.model_path = model_path
        mergeable_ranks = load_tiktoken_bpe(self.model_path)
        
        
        self.sp_model = tiktoken.Encoding(
            name="custom_cl128k",
            pat_str=self.CL100K_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens={**all_special_tokens_with_ids},
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)
# -


