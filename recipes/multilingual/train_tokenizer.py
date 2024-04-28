import fire
import os
import sentencepiece as spm

def main(data_file, save_path, vocab_size=16_000, num_threads=8):
    os.makedirs(save_path, exist_ok=True)
    tokenizer_name = os.path.join(save_path, "tokenizer")
    
    spm.SentencePieceTrainer.train(
        input=data_file,
        model_prefix=tokenizer_name,
        vocab_size=vocab_size,
        num_threads=num_threads,
        model_type="bpe",
        max_sentence_length=1073741824,
        shuffle_input_sentence="true",
        character_coverage=1.0,
        hard_vocab_limit="false",
    )

if __name__ == "__main__":
    fire.Fire(main)
