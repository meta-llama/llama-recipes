"""
Code borrowed from https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
"""

import os
import fire
import re
from transformers import LlamaTokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def main(new_tokenizer_path, extended_tokenizer_save_path):
    original_tokenizer_path = hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="tokenizer.model", local_dir="original_tokenizer")
    original_tokenizer_spm = sp_pb2_model.ModelProto()
    original_tokenizer_spm.ParseFromString(open(original_tokenizer_path, "rb").read())
    new_tokenizer_spm = sp_pb2_model.ModelProto()
    new_tokenizer_spm.ParseFromString(open(os.path.join(new_tokenizer_path, "tokenizer.model"), "rb").read())

    def contains_eng(text):
        eng_pattern = re.compile(r"[\u0020-\u007E]+")
        return True if eng_pattern.search(text) else False

    original_tokenizer_tokenset = set(p.piece for p in original_tokenizer_spm.pieces)
    print(f"Number of tokens before merge: {len(original_tokenizer_tokenset)}")
    for p in new_tokenizer_spm.pieces:
        piece = p.piece
        if piece not in original_tokenizer_tokenset and not contains_eng(piece):
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            original_tokenizer_spm.pieces.append(new_p)
    print(f"Number of tokens after merge: {len(original_tokenizer_spm.pieces)}")

    os.makedirs(extended_tokenizer_save_path, exist_ok=True)
    with open(os.path.join(extended_tokenizer_save_path, "tokenizer.model"), "wb") as f:
        f.write(original_tokenizer_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=os.path.join(extended_tokenizer_save_path, "tokenizer.model"), legacy=False)
    tokenizer.save_pretrained(extended_tokenizer_save_path)
    print(f"Tokenizer saved to {extended_tokenizer_save_path}")

    # Verify that the extended tokenizer's English vocab matches with that of the original Llama tokenizer
    tok1 = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tok2 = LlamaTokenizer.from_pretrained(extended_tokenizer_save_path)
    for i in range(len(tok1)):
        assert tok1.convert_ids_to_tokens(i) == tok2.convert_ids_to_tokens(i), f"Token mismatch at index {i}."


if __name__ == "__main__":
    fire.Fire(main)