import re
import pandas as pd

uyir_vowels = ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ']
nedil_vowels = ['ஆ', 'ஈ', 'ஊ', 'ஏ', 'ஐ', 'ஓ', 'ஔ']
kuril_vowels = ['அ', 'இ', 'உ', 'எ', 'ஒ']

mei_consonants = [
    'க', 'ச', 'ட', 'த', 'ப', 'ற',
    'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன', 
    'ய', 'ர', 'ல', 'வ', 'ழ', 'ள'  
]
vowel_signs = ['', 'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ']
nedil_signs = ['ா', 'ீ', 'ூ', 'ே', 'ோ', 'ை', 'ௌ']
kuril_signs = ['', 'ி', 'ு', 'ெ', 'ொ']
short_vowel_signs = ['ி', 'ு', 'ெ', 'ொ']

def generate_tamil_compounds():
    global tamil_all_compounds, tamil_kuril_compounds, tamil_nedil_compounds
    tamil_all_compounds = []
    tamil_kuril_compounds = []
    tamil_nedil_compounds = []

    for mei in mei_consonants:
        mei_base = mei + '்'
        for nedil in nedil_signs:
            compound = mei + nedil
            tamil_nedil_compounds.append(compound)
        for kuril in kuril_signs:
            if kuril == '':
                compound = mei
                tamil_kuril_compounds.append(compound)
            else:
                compound = mei + kuril
                tamil_kuril_compounds.append(compound)
        for sign in vowel_signs:
            if sign == '':
                compound = mei
            else:
                compound = mei + sign
            tamil_all_compounds.append(compound)

def extract_tamil_pos_data(file_path, output_csv='tamil_pos_dataset.csv', unique_only=False):
    pattern = r'\(([^)]+)\)'
    tamil_words = []
    pos_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            matches = re.findall(pattern, line)
            for match in matches:
                if len(match.split(',')) >= 2:
                    lemma = match.split(',')[0].strip().split('(')[-1]
                    pos_tag = match.split(',')[1].strip()    
                    if lemma and pos_tag and not any(char.isdigit() for char in lemma):
                        tamil_words.append(lemma)
                        pos_tags.append(pos_tag)   
    
    df = pd.DataFrame({
        'tamil_word': tamil_words,
        'pos_tag': pos_tags
    })
    
    if unique_only:
        df = df.drop_duplicates()
        print(f"Removed duplicates: {len(tamil_words)} -> {len(df)} entries")
    df = df[~df['pos_tag'].isin(['VOC','OPT','QW','DET','Nf','INT','puncN','UT','unk','punc'])] 
    df['pos_tag'] = df['pos_tag'].apply(lambda tag: 'ADJ' if 'ADJ' in tag else tag)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Dataset saved to '{output_csv}'")
    print(f"Total entries: {df.shape[0]}")
    print(f"\nUnique Tamil words: {df['tamil_word'].nunique()}")
    print(f"Unique POS tags: {df['pos_tag'].nunique()}")
    print(f"\nMost common POS tags:")
    print(df['pos_tag'].value_counts().head(10))  
    return df

def get_unique_words_only(df):
    unique_words = df['tamil_word'].unique()
    return pd.Series(unique_words, name='tamil_word')

def get_unique_word_pos_pairs(df):
    return df.drop_duplicates()

def analyze_word_pos_relationships(df):
    word_pos_counts = df.groupby('tamil_word')['pos_tag'].nunique().sort_values(ascending=False)
    
    print("\n=== Word-POS Analysis ===")
    print(f"Words with multiple POS tags: {(word_pos_counts > 1).sum()}")
    print(f"Words with single POS tag: {(word_pos_counts == 1).sum()}")
    
    print("\nTop 10 words with most POS variations:")
    for word, count in word_pos_counts.head(10).items():
        pos_tags = df[df['tamil_word'] == word]['pos_tag'].unique()
        print(f"{word:15} -> {count} tags: {', '.join(pos_tags)}")
    
    return word_pos_counts

def export_unique_variations(df, base_filename='tamil_unique'):
    """
    Export different types of unique datasets
    """
    # 1. Unique words only (no POS tags)
    unique_words = get_unique_words_only(df)
    unique_words.to_csv(f'{base_filename}_words_only.csv', index=False, header=['tamil_word'])
    print(f"\n1. Unique words only: {len(unique_words)} words saved to '{base_filename}_words_only.csv'")
    
    # 2. Unique word-POS combinations
    unique_pairs = get_unique_word_pos_pairs(df)
    unique_pairs.to_csv(f'{base_filename}_word_pos_pairs.csv', index=False)
    print(f"2. Unique word-POS pairs: {len(unique_pairs)} pairs saved to '{base_filename}_word_pos_pairs.csv'")
    
    # 3. Word frequency analysis
    word_freq = df['tamil_word'].value_counts().reset_index()
    word_freq.columns = ['tamil_word', 'frequency']
    word_freq.to_csv(f'{base_filename}_word_frequency.csv', index=False)
    print(f"3. Word frequency: {len(word_freq)} words saved to '{base_filename}_word_frequency.csv'")
    
    # 4. Words with their possible POS tags
    word_pos_groups = df.groupby('tamil_word')['pos_tag'].apply(lambda x: ', '.join(sorted(x.unique()))).reset_index()
    word_pos_groups.columns = ['tamil_word', 'possible_pos_tags']
    word_pos_groups.to_csv(f'{base_filename}_word_all_pos.csv', index=False)
    print(f"4. Words with all POS tags: {len(word_pos_groups)} words saved to '{base_filename}_word_all_pos.csv'")
    
    return {
        'unique_words': unique_words,
        'unique_pairs': unique_pairs,
        'word_frequency': word_freq,
        'word_pos_groups': word_pos_groups
    }

def filter_by_frequency(df, min_frequency=2):
    """
    Filter words that appear at least min_frequency times
    """
    word_counts = df['tamil_word'].value_counts()
    frequent_words = word_counts[word_counts >= min_frequency].index
    filtered_df = df[df['tamil_word'].isin(frequent_words)]
    
    print(f"\nFiltered by frequency >= {min_frequency}:")
    print(f"Words: {len(frequent_words)} (from {df['tamil_word'].nunique()})")
    print(f"Total entries: {len(filtered_df)} (from {len(df)})")
    
    return filtered_df

def check_if_lemma(word, word_list):
    for other_word in word_list:
        if (len(other_word) < len(word)) and (other_word in word):
            return False
            break
    return True

def add_aa(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-1] + 'ா'
    else:
        inflected_word = word[:-1] + 'ா'
    return inflected_word

def add_I(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-2] + 'த்தை'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ை'
    elif word[-1] in ['ி', 'ீ', 'ை', 'ு', 'இ', 'ஈ', 'ஐ', 'உ']:
        inflected_word = word + 'யை'
    elif word[-1] in ['', 'ா', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ','ஏ','ே']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ை'
        else:
            inflected_word = word + 'வை'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ை'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ை'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ை'
    else:
        inflected_word = word[:-1] +'ை'
    return inflected_word

def add_Ipatri(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-2] + 'த்தைப்பற்றி'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ைப்பற்றி'
    elif word[-1] in ['ி', 'ீ', 'ை', 'ு', 'இ', 'ஈ', 'ஐ', 'உ']:
        inflected_word = word + 'யைப்பற்றி'
    elif word[-1] in ['', 'ா', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ','ஏ','ே']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ைப்பற்றி'
        else:
            inflected_word = word + 'வைப்பற்றி'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ைப்பற்றி'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ைப்பற்றி'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ைப்பற்றி'
    else:
        inflected_word = word[:-1] +'ைப்பற்றி'
    return inflected_word

def add_kal(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-2] + 'ங்கள்'
    elif word.endswith('ல்'):
        inflected_word = word[:-2] + 'ற்கள்'
    elif word.endswith('ள்'):
        inflected_word = word[:-2] + 'ட்கள்'
    elif word.endswith('ர்'):
        inflected_word = word + 'கள்'
    elif (len(word) <= 2) and ((word in tamil_all_compounds) or (word in mei_consonants) or (word in uyir_vowels)):
        inflected_word = word + 'க்கள்'
    elif (word[:2] in tamil_nedil_compounds) or (word[:2] in nedil_vowels):
        inflected_word = word + 'க்கள்'
    else:
        inflected_word = word + 'கள்'
    return inflected_word
    
def add_in(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்தின்'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ின்'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ின்'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யின்'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ின்'
        else:
            inflected_word = word + 'வின்'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யின்'
        inflected_word2 = word + 'வின்'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ின்'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ின்'
    else:
        inflected_word = word[:-1] + 'ின்'
    return inflected_word, inflected_word2

def add_oodu(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்தோடு'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ோடு'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ோடு'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யோடு'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ோடு'
        else:
            inflected_word = word + 'வோடு'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யோடு'
        inflected_word2 = word + 'வோடு'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ோடு'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ோடு'
    else:
        inflected_word = word[:-1] + 'ோடு'
    return inflected_word, inflected_word2

def add_aaga(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ாக'
        inflected_word2 = word[:-1] + 'ாக'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யாக'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ாக'
        else:
            inflected_word = word + 'வாக'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யாக'
        inflected_word2 = word + 'வாக'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ாக'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ாக'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ாக'
    else:
        inflected_word = word[:-1] + 'ாக'
    return inflected_word, inflected_word2

def add_aana(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ான'
        inflected_word2 = word[:-1] + 'ான'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யான'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ான'
        else:
            inflected_word = word + 'வான'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யான'
        inflected_word2 = word + 'வான'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ான'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ான'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ான'
    else:
        inflected_word = word[:-1] + 'ான'
    return inflected_word, inflected_word2

def add_udan(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்துடன்'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ுடன்'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ுடன்'
        inflected_word2 = word[:-1] + 'ுடன்'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யுடன்'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ுடன்'
        else:
            inflected_word = word + 'வுடன்'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யுடன்'
        inflected_word2 = word + 'வுடன்'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ுடன்'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ுடன்'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ுடன்'
    else:
        inflected_word = word[:-1] +'ுடன்'
    return inflected_word, inflected_word2

def add_illaamal(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ள்'):
        inflected_word = word[:-1]+'ில்லாமல்'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யில்லாமல்'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ில்லாமல்'
        else:
            inflected_word = word + 'வில்லாமல்'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யில்லாமல்'
        inflected_word2 = word + 'வில்லாமல்'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ில்லாமல்'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ில்லாமல்'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ில்லாமல்'
    else:
        inflected_word = word[:-1] +'ில்லாமல்'
    return inflected_word, inflected_word2

def add_idam(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்திடம்'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ிடம்'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யிடம்'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ிடம்'
        else:
            inflected_word = word + 'விடம்'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யிடம்'
        inflected_word2 = word + 'விடம்'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ிடம்'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ிடம்'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ிடம்'
    else:
        inflected_word = word[:-1] +'ிடம்'
    return inflected_word, inflected_word2

def add_il(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்தில்'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ில்'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யில்'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ில்'
        else:
            inflected_word = word + 'வில்'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யில்'
        inflected_word2 = word + 'வில்'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ில்'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ில்'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ில்'
    else:
        inflected_word = word[:-1] +'ில்'
    return inflected_word, inflected_word2

def add_ilirundhu(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்திலிருந்து'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ிலிருந்து'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யிலிருந்து'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ிலிருந்து'
        else:
            inflected_word = word + 'விலிருந்து'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யிலிருந்து'
        inflected_word2 = word + 'விலிருந்து'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ிலிருந்து'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ிலிருந்து'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ிலிருந்து'
    else:
        inflected_word = word[:-1] +'ிலிருந்து'
    return inflected_word, inflected_word2

def add_ukku(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்துக்கு'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ுக்கு'
    elif word[-1] in ['ி', 'ீ', 'ை', 'ு', 'இ', 'ஈ', 'ஐ', 'உ']:
        inflected_word = word + 'க்கு'
    elif word[-1] in ['', 'ா', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ','ஏ','ே']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ுக்கு'
        else:
            inflected_word = word + 'வுக்கு'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ுக்கு'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ுக்கு'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ுக்கு'
    else:
        inflected_word = word[:-1] +'ுக்கு'
    return inflected_word, inflected_word2

def add_adhu(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்தது'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'து'
    elif word[-1] in ['ி', 'ீ', 'ை', 'ு', 'இ', 'ஈ', 'ஐ', 'உ']:
        inflected_word = word + 'யது'
    elif word[-1] in ['', 'ா', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ','ஏ','ே']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'து'
        else:
            inflected_word = word + 'வது'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'து'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'து'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'து'
    else:
        inflected_word = word[:-1] +'து'
    return inflected_word

def add_udaiya(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-2]+'த்துடைய'
    elif word.endswith('டு') or word.endswith('று'):
        inflected_word = word[:-1]+'்'+word[-2]+'ுடைய'
    elif word[-1] in ['ி', 'ீ', 'ை', 'ு', 'இ', 'ஈ', 'ஐ', 'உ']:
        inflected_word = word + 'யுடைய'
    elif word[-1] in ['', 'ா', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ','ஏ','ே']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ுடைய'
        else:
            inflected_word = word + 'வுடைய'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ுடைய'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ுடைய'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ுடைய'
    else:
        inflected_word = word[:-1] +'ுடைய'
    return inflected_word

def add_irundhu(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-1]+'ிருந்து'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+word[-2]+'ிருந்து'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யிருந்து'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ிருந்து'
        else:
            inflected_word = word + 'விருந்து'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யிருந்து'
        inflected_word2 = word + 'விருந்து'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ிருந்து'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ிருந்து'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ிருந்து'
    else:
        inflected_word = word[:-1] +'ிருந்து'
    return inflected_word, inflected_word2

def add_aal(word, suffix):
    inflected_word2 = 'null'
    if word.endswith('ம்'):
        inflected_word = word[:-1]+'த்தால்'
    elif word.endswith('ள்'):
        inflected_word = word[:-1]+'ால்'
    elif word[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = word + 'யால்'
    elif word[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (word[-1] == 'ு') and (len(word)>=4) and (word[-2] == word[-4]):
            inflected_word = word[:-1]+'ால்'
        else:
            inflected_word = word + 'வால்'
    elif word[-1] in ['ஏ','ே']:
        inflected_word = word + 'யால்'
        inflected_word2 = word + 'வால்'
    elif len(word) ==4 and word[-1] == '்' and word[1] in short_vowel_signs and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ால்'
    elif len(word)<=3 and word[-1] == '்' and word[0] in mei_consonants:
        inflected_word = word+word[-2]+'ால்'
    elif word[-1] == '்':
        inflected_word = word[:-1]+'ால்'
    else:
        inflected_word = word[:-1] +'ால்'
    return inflected_word, inflected_word2

def add_kalinudan(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-2] + 'ங்களினுடன்'
    elif word.endswith('ல்'):
        inflected_word = word[:-2] + 'ற்களினுடன்'
    elif word.endswith('ள்'):
        inflected_word = word[:-2] + 'ட்களினுடன்'
    elif word.endswith('ர்'):
        inflected_word = word + 'களினுடன்'
    elif (len(word) <= 2) and ((word in tamil_all_compounds) or (word in mei_consonants) or (word in uyir_vowels)):
        inflected_word = word + 'க்களினுடன்'
    elif (word[:2] in tamil_nedil_compounds) or (word[:2] in nedil_vowels):
        inflected_word = word + 'க்களினுடன்'
    else:
        inflected_word = word + 'களினுடன்'
    return inflected_word

def add_ukkaaga(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-2] + 'ங்களுக்காக'
    elif word.endswith('ல்'):
        inflected_word = word[:-2] + 'ற்களுக்காக'
    elif word.endswith('ள்'):
        inflected_word = word[:-2] + 'ட்களுக்காக'
    elif word.endswith('ர்'):
        inflected_word = word + 'களுக்காக'
    elif (len(word) <= 2) and ((word in tamil_all_compounds) or (word in mei_consonants) or (word in uyir_vowels)):
        inflected_word = word + 'க்களுக்காக'
    elif (word[:2] in tamil_nedil_compounds) or (word[:2] in nedil_vowels):
        inflected_word = word + 'க்களுக்காக'
    else:
        inflected_word = word + 'களுக்காக'
    return inflected_word