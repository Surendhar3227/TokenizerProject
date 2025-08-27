# Core_functions.py

from tokenizers import Tokenizer
import sentencepiece as spm
from transformers import AutoTokenizer
import pandas as pd
import re

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True, token=access_token)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
mbart_tokenizer = tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
indic_bert = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=access_token)

def obtain_tokenizers(vocab_size):
    if vocab_size == "3k":
        bpe_path = './GPU_Tokenizer/3/tamil_sentencepiece_bpe_3k.model'
        unigram_path = './GPU_Tokenizer/3/tamil_sentencepiece_unigram_3k.model'
        tok_bpe = spm.SentencePieceProcessor()
        tok_bpe.load(bpe_path)
        tok_unigram = spm.SentencePieceProcessor()
        tok_unigram.load(unigram_path)
        tok_byte_bpe = Tokenizer.from_file("./GPU_Tokenizer/3/tamil_byte_level_bpe_3k.json")
        return tok_bpe, tok_unigram, tok_byte_bpe

    elif vocab_size == "5k":
        bpe_path = './GPU_Tokenizer/5/tamil_sentencepiece_bpe_5k.model'
        unigram_path = './GPU_Tokenizer/5/tamil_sentencepiece_unigram_5k.model'
        tok_bpe = spm.SentencePieceProcessor()
        tok_bpe.load(bpe_path)
        tok_unigram = spm.SentencePieceProcessor()
        tok_unigram.load(unigram_path)
        tok_byte_bpe = Tokenizer.from_file("./GPU_Tokenizer/5/tamil_byte_level_bpe_5k.json")
        return tok_bpe, tok_unigram, tok_byte_bpe
    
    elif vocab_size == "8k":
        bpe_path = './GPU_Tokenizer/8/tamil_sentencepiece_bpe_8k.model'
        unigram_path = './GPU_Tokenizer/8/tamil_sentencepiece_unigram_8k.model'
        tok_bpe = spm.SentencePieceProcessor()
        tok_bpe.load(bpe_path)
        tok_unigram = spm.SentencePieceProcessor()
        tok_unigram.load(unigram_path)
        tok_byte_bpe = Tokenizer.from_file("./GPU_Tokenizer/8/tamil_byte_level_bpe_8k.json")
        return tok_bpe, tok_unigram, tok_byte_bpe
        
    elif vocab_size == "10k":
        bpe_path = './GPU_Tokenizer/10/tamil_sentencepiece_bpe_10k.model'
        unigram_path = './GPU_Tokenizer/10/tamil_sentencepiece_unigram_10k.model'
        tok_bpe = spm.SentencePieceProcessor()
        tok_bpe.load(bpe_path)
        tok_unigram = spm.SentencePieceProcessor()
        tok_unigram.load(unigram_path)
        tok_byte_bpe = Tokenizer.from_file("./GPU_Tokenizer/10/tamil_byte_level_bpe_10k.json")
        return tok_bpe, tok_unigram, tok_byte_bpe

    elif vocab_size == "20k":
        bpe_path = './GPU_Tokenizer/20/tamil_sentencepiece_bpe_20k.model'
        unigram_path = './GPU_Tokenizer/20/tamil_sentencepiece_unigram_20k.model'
        tok_bpe = spm.SentencePieceProcessor()
        tok_bpe.load(bpe_path)
        tok_unigram = spm.SentencePieceProcessor()
        tok_unigram.load(unigram_path)
        tok_byte_bpe = Tokenizer.from_file("./GPU_Tokenizer/20/tamil_byte_level_bpe_20k.json")
        return tok_bpe, tok_unigram, tok_byte_bpe

    elif vocab_size == "32k":
        bpe_path = './GPU_Tokenizer/32/tamil_sentencepiece_bpe_32k.model'
        unigram_path = './GPU_Tokenizer/32/tamil_sentencepiece_unigram_32k.model'
        tok_bpe = spm.SentencePieceProcessor()
        tok_bpe.load(bpe_path)
        tok_unigram = spm.SentencePieceProcessor()
        tok_unigram.load(unigram_path)
        tok_byte_bpe = Tokenizer.from_file("./GPU_Tokenizer/32/tamil_byte_level_bpe_32k.json")
        return tok_bpe, tok_unigram, tok_byte_bpe

def remove_sentencepiece_underscore(tokens):
    return [token.replace('▁', '') if token.startswith('▁') else token for token in tokens]
    
def remove_bert_characters(tokens):
    return [token.replace('##', '') if token.startswith('#') else token for token in tokens]

def tokenize_dataset(Corpus_df, output_file: str, vocab_size: str, sentence_column: str, skip_saving=True, Type='Sentence'):
    bpe_tokenizer, unigram_tokenizer, byte_tokenizer = obtain_tokenizers(vocab_size)
    test_sentence_list = Corpus_df[sentence_column].to_list()
    tokenized_df = pd.DataFrame()
    tokenized_df['Sentence'] = test_sentence_list
    if Type == 'Sentence':
        tokenized_df['Sentence Length'] = [len(sentence.split()) for sentence in tokenized_df['Sentence']]
    tokenized_df['BPE Tokenized'] = [bpe_tokenizer.encode_as_pieces(sent) for sent in test_sentence_list]
    tokenized_df['BPE Tokenized'] = tokenized_df['BPE Tokenized'].apply(remove_sentencepiece_underscore)
    tokenized_df['Unigram Tokenized'] = [unigram_tokenizer.encode_as_pieces(sent) for sent in test_sentence_list]
    tokenized_df['Unigram Tokenized'] = tokenized_df['Unigram Tokenized'].apply(remove_sentencepiece_underscore)
    tokenized_df['Byte-BPE Tokenized'] = [(byte_tokenizer.encode(sent)).tokens for sent in test_sentence_list]
    tokenized_df['Token Length BPE'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['BPE Tokenized']]
    tokenized_df['Token Length Unigram'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['Unigram Tokenized']]
    tokenized_df['Token Length Byte-BPE'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['Byte-BPE Tokenized']]
    if not skip_saving:
        tokenized_df.to_csv(output_file)
    return tokenized_df

def tokenize_dataset_existing(Corpus_df, output_file: str, sentence_column: str, skip_saving=True, gemma_tokenizer = gemma_tokenizer, 
                              bert_tokenizer=bert_tokenizer, xlmr_tokenizer=xlmr_tokenizer, mt5_tokenizer=mt5_tokenizer, mbart = mbart_tokenizer, indic = indic_bert):
    test_sentence_list = Corpus_df[sentence_column].to_list()
    tokenized_df = pd.DataFrame()
    tokenized_df['Words'] = test_sentence_list
    tokenized_df['Gemma Tokenized'] = [gemma_tokenizer.tokenize(sent) for sent in test_sentence_list]
    tokenized_df['mBERT Tokenized'] = [bert_tokenizer.tokenize(sent) for sent in test_sentence_list]
    tokenized_df['mBERT Tokenized'] = tokenized_df['mBERT Tokenized'].apply(remove_bert_characters)
    tokenized_df['XLMR Tokenized'] = [xlmr_tokenizer.tokenize(sent) for sent in test_sentence_list]
    tokenized_df['XLMR Tokenized'] = tokenized_df['XLMR Tokenized'].apply(remove_sentencepiece_underscore)
    tokenized_df['mt5 Tokenized'] = [mt5_tokenizer.tokenize(sent) for sent in test_sentence_list]
    tokenized_df['mt5 Tokenized'] = tokenized_df['mt5 Tokenized'].apply(remove_sentencepiece_underscore)
    tokenized_df['mBART Tokenized'] = [mbart.tokenize(sent) for sent in test_sentence_list]
    tokenized_df['mBART Tokenized'] = tokenized_df['mBART Tokenized'].apply(remove_sentencepiece_underscore)
    tokenized_df['IndicBERT Tokenized'] = [indic_bert.tokenize(sent) for sent in test_sentence_list]
    tokenized_df['IndicBERT Tokenized'] = tokenized_df['IndicBERT Tokenized'].apply(remove_sentencepiece_underscore)
    tokenized_df['Token Length Gemma'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['Gemma Tokenized']]
    tokenized_df['Token Length mBERT'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['mBERT Tokenized']]
    tokenized_df['Token Length XLMR'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['XLMR Tokenized']]
    tokenized_df['Token Length mt5'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['mt5 Tokenized']]
    tokenized_df['Token Length mBART'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['mBART Tokenized']]
    tokenized_df['Token Length IndicBERT'] = [len(tokenized_sentence) for tokenized_sentence in tokenized_df['IndicBERT Tokenized']]
    if not skip_saving:
        tokenized_df.to_csv(output_file)
    return tokenized_df

def df_extract_unique_words(df, column='sentence_tam'):
    unique_words = set()    
    punctuation = r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~।॥၊၊॥‘’“”…'
    batch_sentences = df[column].astype(str)
    batch_sentences = batch_sentences.str.translate(str.maketrans('', '', punctuation))
    batch_text = ' '.join(batch_sentences)
    batch_words = [words.strip() for words in batch_text.split()]
    batch_unique_words = set(batch_words)
    unique_words.update(batch_unique_words)
    unique_word_list = list(unique_words)
    return(pd.DataFrame(unique_word_list))