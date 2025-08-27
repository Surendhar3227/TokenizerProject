import re
import os
import sys
import json
import unicodedata
from typing import List, Tuple, Iterator
from collections import Counter
import sentencepiece as spm
from tokenizers import Tokenizer
import pandas as pd
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TamilTokenizerTrainer:
    def __init__(self, output_dir: str = "./tokenizers"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def train_sentencepiece_bpe(self, input_file: str, vocab_size: int = 16000, 
                               num_threads: int = 4) -> str:
        model_prefix = os.path.join(self.output_dir, "/20K/tamil_sentencepiece_bpe")       
        logger.info(f"Training SentencePiece BPE tokenizer with vocab_size={vocab_size}")
        spm.SentencePieceTrainer.Train(
            input=input_file,
            input_format='text',
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            num_threads=num_threads * 2,
            split_by_whitespace=True,
            split_by_number=True,
            max_sentencepiece_length=16,
            shuffle_input_sentence=True,
            input_sentence_size=2000000,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            train_extremely_large_corpus=True,
            user_defined_symbols=['<mask>'],
            normalization_rule_name='nmt_nfkc_cf'
            # byte_fallback=True 
        )
        
        logger.info(f"SentencePiece BPE model saved: {model_prefix}.model")
        return f"{model_prefix}.model"
    
    def train_sentencepiece_unigram(self, input_file: str, vocab_size: int = 16000,
                                  num_threads: int = 4) -> str:
        model_prefix = os.path.join(self.output_dir, "/20K/tamil_sentencepiece_unigram")       
        logger.info(f"Training SentencePiece Unigram tokenizer with vocab_size={vocab_size}")     
        spm.SentencePieceTrainer.Train(
            input=input_file,
            input_format='text',
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='unigram', 
            character_coverage=1.0,
            num_threads=num_threads,
            split_by_whitespace=True,
            max_sentencepiece_length=16,
            shuffle_input_sentence=True,
            input_sentence_size=2000000,
            remove_extra_whitespaces=True,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            train_extremely_large_corpus=True,
            user_defined_symbols=['<mask>'],
            normalization_rule_name='nmt_nfkc_cf'
        )
        
        logger.info(f"SentencePiece Unigram model saved: {model_prefix}.model")
        return f"{model_prefix}.model"
    
    def train_byte_level_bpe(self, input_file: str, vocab_size: int = 16000) -> str:
        logger.info(f"Training Byte-level BPE tokenizer with vocab_size={vocab_size}")
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
            show_progress=True
        )
        def text_iterator():
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        output_path = os.path.join(self.output_dir, "tamil_byte_level_bpe.json")
        tokenizer.save(output_path)
        
        logger.info(f"Byte-level BPE tokenizer saved: {output_path}")
        return output_path
    
def main():
    INPUT_FILE = "./Datasets/IndicNLP_Tamil_sentences.txt" 
    VOCAB_SIZE = 20000
    NUM_THREADS = os.cpu_count() 
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file {INPUT_FILE} not found!")
        logger.info("Please ensure your Tamil dataset file is named 'tamil_dataset.txt'")
        return
    
    logger.info("=== Training Tokenizers ===")
    trainer = TamilTokenizerTrainer()
    
    # Train all three tokenizers
    sp_bpe_model = trainer.train_sentencepiece_bpe(INPUT_FILE, VOCAB_SIZE, NUM_THREADS)  
    sp_unigram_model = trainer.train_sentencepiece_unigram(INPUT_FILE, VOCAB_SIZE, NUM_THREADS)
    byte_bpe_model = trainer.train_byte_level_bpe(INPUT_FILE, VOCAB_SIZE)
    
    logger.info("=== Training Complete ===")
    logger.info(f"Training Dataset: {INPUT_FILE}")
    logger.info(f"SentencePiece BPE: {sp_bpe_model}")
    logger.info(f"SentencePiece Unigram: {sp_unigram_model}")
    logger.info(f"Byte-level BPE: {byte_bpe_model}")

if __name__ == "__main__":
    main()