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
    suffixes = ['கள்','இன்','ஆக','ஆன','உடன்','இல்லாமல்','இடம்','இலிருந்து','இல்']
    non_lemma_words = []
    lemma_words = []

    inflected_words = []
    for noun in noun_list:
        noun=noun.strip()
        for suffix in suffixes:
            if suffix == 'கள்':
                inflected_word = add_kal(noun, suffix)
                inflected_words.append(inflected_word)
            
            elif suffix == 'இன்':
                inflected_word1, inflected_word2 = add_in(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)
                kal_added_noun = add_kal(noun, 'கள்')
                inflected_word1, inflected_word2 = add_in(kal_added_noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'ஆக':
                inflected_word1, inflected_word2 = add_aaga(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'ஆன':
                inflected_word1, inflected_word2 = add_aana(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'உடன்':
                inflected_word1, inflected_word2 = add_udan(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)
                kal_added_noun = add_kal(noun, 'கள்')
                inflected_word1, inflected_word2 = add_udan(kal_added_noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'இல்லாமல்':
                inflected_word1, inflected_word2 = add_illaamal(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)
                kal_added_noun = add_kal(noun, 'கள்')
                inflected_word1, inflected_word2 = add_illaamal(kal_added_noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'இடம்':
                inflected_word1, inflected_word2 = add_idam(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)
                kal_added_noun = add_kal(noun, 'கள்')
                inflected_word1, inflected_word2 = add_idam(kal_added_noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'இல்':
                inflected_word1, inflected_word2 = add_il(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)
                kal_added_noun = add_kal(noun, 'கள்')
                inflected_word1, inflected_word2 = add_idam(kal_added_noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

            elif suffix == 'இலிருந்து':
                inflected_word1, inflected_word2 = add_ilirundhu(noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)
                kal_added_noun = add_kal(noun, 'கள்')
                inflected_word1, inflected_word2 = add_ilirundhu(kal_added_noun, suffix)
                inflected_words.append(inflected_word1)
                if inflected_word2 != 'null':
                    inflected_words.append(inflected_word2)

                        
    print(inflected_words[:20])