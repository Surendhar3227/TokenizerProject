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

def add_kal(word, suffix):
    if word.endswith('ம்'):
        inflected_word = word[:-2] + 'ங்கள்'
    elif word.endswith('ல்'):
        inflected_word = word[:-2] + 'ற்கள்'
    elif word.endswith('ள்'):
        inflected_word = word[:-2] + 'ட்கள்'
    elif (len(word) <= 2) and ((word in tamil_all_compounds) or (word in mei_consonants) or (word in uyir_vowels)):
        inflected_word = word + 'க்கள்'
    elif (word[:2] in tamil_nedil_compounds) or (word[:2] in nedil_vowels):
        inflected_word = word + 'க்கள்'
    else:
        inflected_word = word + 'கள்'
    return inflected_word

def add_in(word, suffix):
    noun = word
    inflected_word2 = 'null'
    if noun.endswith('ம்'):
        inflected_word = noun[:-2]+'த்தின்'
    elif noun.endswith('ள்'):
        inflected_word = noun[:-1]+'ின்'
    elif noun.endswith('டு') or noun.endswith('று'):
        inflected_word = noun[:-1]+'்'+noun[-2]+'ின்'
    elif noun[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = noun + 'யின்'
    elif noun[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (noun[-1] == 'ு') and (len(noun)>=4) and (noun[-2] == noun[-4]):
            inflected_word = noun[:-1]+'ின்'
        else:
            inflected_word = noun + 'வின்'
    elif noun[-1] in ['ஏ','ே']:
        inflected_word = noun + 'யின்'
        inflected_word2 = noun + 'வின்'
    elif noun[-1] == '்' and (noun[-3] in short_vowel_signs or noun[-3] in mei_consonants):
        inflected_word = noun+noun[-2]+'ின்'
    else:
        inflected_word = noun[:-1] + 'ின்'

    return inflected_word, inflected_word2

def add_aaga(word, suffix):
    noun = word
    inflected_word2 = 'null'
    if noun.endswith('டு') or noun.endswith('று'):
        inflected_word = noun[:-1]+'்'+noun[-2]+'ாக'
        inflected_word2 = noun[:-1] + 'ாக'
    elif noun[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = noun + 'யாக'
    elif noun[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (noun[-1] == 'ு') and (len(noun)>=4) and (noun[-2] == noun[-4]):
            inflected_word = noun[:-1]+'ாக'
        else:
            inflected_word = noun + 'வாக'
    elif noun[-1] in ['ஏ','ே']:
        inflected_word = noun + 'யாக'
        inflected_word2 = noun + 'வாக'
    elif noun[-1] == '்' and (noun[-3] in short_vowel_signs or noun[-3] in mei_consonants):
        inflected_word = noun+noun[-2]+'ாக'
    elif noun[-1] == '்':
        inflected_word = noun[:-1]+'ாக'
    else:
        inflected_word = noun[:-1] + 'ாக'
    return inflected_word, inflected_word2

def add_aana(word, suffix):
    noun = word
    inflected_word2 = 'null'
    if noun.endswith('டு') or noun.endswith('று'):
        inflected_word = noun[:-1]+'்'+noun[-2]+'ான'
        inflected_word2 = noun[:-1] + 'ான'
    elif noun[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = noun + 'யான'
    elif noun[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (noun[-1] == 'ு') and (len(noun)>=4) and (noun[-2] == noun[-4]):
            inflected_word = noun[:-1]+'ான'
        else:
            inflected_word = noun + 'வான'
    elif noun[-1] in ['ஏ','ே']:
        inflected_word = noun + 'யான'
        inflected_word2 = noun + 'வான'
    elif noun[-1] == '்' and (noun[-3] in short_vowel_signs or noun[-3] in mei_consonants):
        inflected_word = noun+noun[-2]+'ான'
    elif noun[-1] == '்':
        inflected_word = noun[:-1]+'ான'
    else:
        inflected_word = noun[:-1] + 'ான'
    return inflected_word, inflected_word2

def add_udan(word, suffix):
    noun = word
    inflected_word2 = 'null'
    if noun.endswith('ம்'):
        inflected_word = noun[:-2]+'த்துடன்'
    elif noun.endswith('ள்'):
        inflected_word = noun[:-1]+'ுடன்'
    elif noun.endswith('டு') or noun.endswith('று'):
        inflected_word = noun[:-1]+'்'+noun[-2]+'ுடன்'
        inflected_word2 = noun[:-1] + 'ுடன்'
    elif noun[-1] in ['ி', 'ீ', 'ை', 'இ', 'ஈ', 'ஐ']:
        inflected_word = noun + 'யுடன்'
    elif noun[-1] in ['', 'ா', 'ு', 'ூ', 'ெ', 'ொ', 'ோ', 'ௌ', 'அ', 'ஆ', 'உ', 'ஊ', 'எ', 'ஒ', 'ஓ', 'ஔ']:
        if (noun[-1] == 'ு') and (len(noun)>=4) and (noun[-2] == noun[-4]):
            inflected_word = noun[:-1]+'ுடன்'
        else:
            inflected_word = noun + 'வுடன்'
    elif noun[-1] in ['ஏ','ே']:
        inflected_word = noun + 'யுடன்'
        inflected_word2 = noun + 'வுடன்'
    elif noun[-1] == '்' and (noun[-3] in short_vowel_signs or noun[-3] in mei_consonants):
        inflected_word = noun+noun[-2]+'ுடன்'
    elif noun[-1] == '்':
        inflected_word = noun[:-1]+'ுடன்'
    else:
        inflected_word = noun[:-1] +'ுடன்'
    return inflected_word, inflected_word2