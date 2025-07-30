import pandas as pd
from UserFunctions import *

if __name__ == "__main__":
    generate_tamil_compounds()
    input_file = 'Datasets\DAILYTHANTHI.txt'
    print("=== Extracting all data ===")
    df = extract_tamil_pos_data(input_file, 'tamil_pos_all.csv')

    print("\n=== Extracting unique combinations ===")
    df_unique = extract_tamil_pos_data(input_file, 'tamil_pos_unique.csv', unique_only=True)

    word_pos_analysis = analyze_word_pos_relationships(df)

    print("\n=== Exporting unique variations ===")
    unique_datasets = export_unique_variations(df)

    unique_words_list = df['tamil_word'].unique().tolist()
    print(f"\nUnique words as list: {len(unique_words_list)} words")
    print("First 10 unique words:", unique_words_list[:10])

    frequent_df = filter_by_frequency(df, min_frequency=3)

    ambiguous_words = df.groupby('tamil_word')['pos_tag'].nunique()
    multi_pos_words = ambiguous_words[ambiguous_words > 1].index.tolist()
    print(f"\nWords with multiple POS tags: {len(multi_pos_words)}")
    print("Examples:", multi_pos_words[:5])

    Unique_pos_df = pd.read_csv('Outputs\\tamil_pos_unique.csv')
    Unique_pos_df.head()
    Unique_noun_df = Unique_pos_df[Unique_pos_df['pos_tag'] == 'N']
    Unique_noun_df.tail(5)

    noun_list = Unique_noun_df['tamil_word'].tolist()
    suffixes = ['கள்','இன்','ஆக','ஆன','உடன்','இல்லாமல்','இடம்','களினுடன்','களுக்காக','ஐப்பற்றி',
                'இலிருந்து','இல்','உக்கு','ஆ','அது', 'உடைய', 'ஓடு','ஐ','இருந்து','ஆல்']
    non_lemma_words = []
    lemma_words = []

    inflected_words = []
    morphological_list =[]
    for noun in noun_list:
        noun=noun.strip()
        for suffix in suffixes:
            match suffix:
                case 'கள்':
                    inflected_word = add_kal(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + '+suffix)
                case 'களினுடன்':
                    inflected_word = add_kalinudan(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + கள் + இன் + உடன்')
                case 'களுக்காக':
                    inflected_word = add_ukkaaga(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + கள் + உக்கு + ஆக')
                case 'இன்':
                    inflected_word1, inflected_word2 = add_in(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_in(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'ஆக':
                    inflected_word1, inflected_word2 = add_aaga(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                case 'ஆன':
                    inflected_word1, inflected_word2 = add_aana(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                case 'உடன்':
                    inflected_word1, inflected_word2 = add_udan(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_udan(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'இல்லாமல்':
                    inflected_word1, inflected_word2 = add_illaamal(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_illaamal(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2) 
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'இடம்':
                    inflected_word1, inflected_word2 = add_idam(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_idam(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2) 
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'இல்':
                    inflected_word1, inflected_word2 = add_il(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_il(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'இருந்து':
                    kal_added_noun = add_kal(noun, suffix)
                    branches = [kal_added_noun, noun]
                    for branch in branches:
                        idam_added_word1, idam_added_word2 = add_idam(branch, suffix)
                        inflected_word1, inflected_word2 = add_irundhu(idam_added_word1, suffix)
                        inflected_words.append(inflected_word1)
                        if branch == kal_added_noun:
                            morphological_list.append(noun+' + கள் + இடம் + '+suffix)
                        else:
                            morphological_list.append(noun+' + இடம் + '+suffix)
                        if inflected_word2 != 'null':
                            inflected_words.append(inflected_word2)
                            if branch == kal_added_noun:
                                morphological_list.append(noun+' + கள் + இடம் + '+suffix)
                            else:
                                morphological_list.append(noun+' + இடம் + '+suffix)                  
                        if idam_added_word2 != 'null':
                            inflected_word1, inflected_word2 = add_irundhu(idam_added_word2, suffix)
                            inflected_words.append(inflected_word1)
                            if branch == kal_added_noun:
                                morphological_list.append(noun+' + கள் + இடம் + '+suffix)
                            else:
                                morphological_list.append(noun+' + இடம் + '+suffix) 
                            if inflected_word2 != 'null':
                                inflected_words.append(inflected_word2)
                                if branch == kal_added_noun:
                                    morphological_list.append(noun+' + கள் + இடம் + '+suffix)
                                else:
                                    morphological_list.append(noun+' + இடம் + '+suffix)              
                case 'இலிருந்து':
                    inflected_word1, inflected_word2 = add_ilirundhu(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_ilirundhu(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2) 
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'உக்கு':
                    inflected_word1, inflected_word2 = add_ukku(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_ukku(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'ஓடு':
                    inflected_word1, inflected_word2 = add_oodu(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_oodu(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + கள் + '+suffix)
                case 'ஆ':
                    inflected_word = add_aa(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word = add_aa(kal_added_noun, suffix)
                    inflected_words.append(inflected_word) 
                    morphological_list.append(noun+' + கள் + '+suffix)
                case 'ஐ':
                    inflected_word = add_I(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word = add_I(kal_added_noun, suffix)
                    inflected_words.append(inflected_word) 
                    morphological_list.append(noun+' + கள் + '+suffix)
                case 'ஐப்பற்றி':
                    inflected_word = add_Ipatri(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + ஐ + ப் + பற்றி + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word = add_Ipatri(kal_added_noun, suffix)
                    inflected_words.append(inflected_word) 
                    morphological_list.append(noun+' + கள் + ஐ + ப் + பற்றி + '+suffix)
                case 'அது':
                    inflected_word = add_adhu(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word = add_adhu(kal_added_noun, suffix)
                    inflected_words.append(inflected_word) 
                    morphological_list.append(noun+' + கள் + '+suffix)
                case 'உடைய':
                    inflected_word = add_udaiya(noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word = add_udaiya(kal_added_noun, suffix)
                    inflected_words.append(inflected_word)
                    morphological_list.append(noun+' + கள் + '+suffix)
                case 'ஆல்':
                    inflected_word1, inflected_word2 = add_aal(noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + '+suffix)
                    kal_added_noun = add_kal(noun, 'கள்')
                    inflected_word1, inflected_word2 = add_aal(kal_added_noun, suffix)
                    inflected_words.append(inflected_word1)
                    morphological_list.append(noun+' + கள் + '+suffix)
                    if inflected_word2 != 'null':
                        inflected_words.append(inflected_word2)
                        morphological_list.append(noun+' + கள் + '+suffix)
                case _:
                        pass  # Do nothing for other cases

    morpho_df = pd.DataFrame({'Word':inflected_words, 'Morphology':morphological_list})
    morpho_df.head()                    
    print(inflected_words[:20])