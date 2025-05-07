import string
import math
from pathlib import Path
from collections import Counter
import re
from wordfreq import word_frequency

def bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY):
    words_in_line = line.split()
    top_k = int(TOP_K_ENTROPY)

    return sorted(words_in_line, key=lambda word: entropy_map.get(word, float('inf')))[:top_k]


def generate_entropy_map_wordfreq(words, lang='en'):
    """
    Generates an entropy map using word frequencies from the `wordfreq` library.

    Args:
        words (Iterable[str]): Collection of words to evaluate.
        lang (str): Language code for the `wordfreq` API (default is 'en').

    Returns:
        dict: Mapping of words to their entropy values.
    """
    entropy_map = {}
    for word in words:
        freq = word_frequency(word, lang)
        if freq > 0:
            entropy_map[word] = -freq * math.log2(freq)
    return entropy_map


def calculate_entropy(word_freq, total_words):
    """
    Calculates Shannon entropy for a word given its frequency.

    Args:
        word_freq (int): Frequency count of the word.
        total_words (int): Total number of words in the dataset.

    Returns:
        float: Calculated entropy value.
    """
    probability = word_freq / total_words
    return -probability * math.log2(probability)


def create_entropy_map(data_sources, mode='books'):
    """
    Creates an entropy map from a dataset or text files based on word frequencies.

    Args:
        data_sources (Iterable[str or dict]): List of text samples or file paths.
        mode (str): Dataset mode, e.g., 'Books', 'Arxiv', 'PILE', etc.

    Returns:
        dict: Mapping of words to entropy values based on frequency.
    """
    word_counts = Counter()

    if mode == 'Arxiv' or mode == 'ECHR':
        print(f"Started processing for mode: {mode}")
        # db_data = procces_data.load_data(mode="Arxiv")
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)
        # for dataset in data_sources:
        #     for item in dataset:
        #         words = item['text'].split()
        #         word_counts.update(words)
    elif mode == 'Books':
        for file_path in data_sources:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    words = line.strip().split()
                    word_counts.update(words)
    elif mode == 'PILE' or mode[:4].lower() == 'pile':
        print(f"Started processing for mode: {mode}")
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)
    else:
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)

    total_words = sum(word_counts.values())
    # print(f"Total words: {total_words}")
    # Generate the entropy map using word frequency
    entropy_map = generate_entropy_map_wordfreq(word_counts.keys())

    return entropy_map


def save_entropy_map(entropy_map, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, entropy in entropy_map.items():
            file.write(f"{word} {entropy}\n")


def load_entropy_map(filename):
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, entropy = line.split()
            entropy_map[word] = float(entropy)
    return entropy_map


def sort_entropy_map(entropy_map, descending=True):
    return sorted(entropy_map.items(), key=lambda item: item[1], reverse=descending)


def get_text_files(folder_path):
    return [str(filepath) for filepath in Path(folder_path).rglob('*.txt')]


def strip_punctuation(word):
    return word.strip(string.punctuation)


def create_line_to_top_words_map(text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY,
                                 nlp_spacy):
    """
    Splits text into sentences and maps each to its top entropy-relevant words.

    Args:
        text (str): Input text to process.
        entropy_map (dict): Word-to-entropy mapping.
        MAX_LEN_LINE_GENERATE (int): Maximum number of words allowed in a sentence.
        MIN_LEN_LINE_GENERATE (int): Minimum number of words required in a sentence.
        TOP_K_ENTROPY (int): Number of low-entropy words to extract per sentence.
        nlp_spacy (spacy.Language): spaCy language model for sentence splitting and NER.

    Returns:
        Tuple[dict, List[str]]: Mapping of sentence index to top entropy words, and the list of valid sentences.
    """
    # text = text.replace('\n', '')
    doc = nlp_spacy(text)
    # Debugging: convert iterator to list to check content
    all_sentences = list(doc.sents)
    sentences = [sent.text.strip() for sent in all_sentences if
                 MIN_LEN_LINE_GENERATE <= len(sent.text.split()) <= MAX_LEN_LINE_GENERATE]

    line_to_top_words_map = {}

    for line_num, line in enumerate(sentences, 1):
        if line.strip():
            top_k_words = {re.sub(r'^\W+|\W+$', '', word.strip(string.punctuation)) for word in
                           bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY) if ' ' not in word}

            ners = {strip_punctuation(ent.text) for ent in doc.ents if ' ' not in ent.text and ent.sent.text == line}
            unique_words = top_k_words.union(ners)
            line_to_top_words_map[line_num] = list(unique_words)

    return line_to_top_words_map, sentences

