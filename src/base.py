import re
import pandas as pd
import heapq
import numpy as np
from typing import List, Dict


def get_cleaned_text(text: str) -> str:
    """Returns cleaned text.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Cleaned text.
    """
    # if text is None or len(str(text)) == 1:
    #     return ''
    stopwords = ['a','the','of','on','in','an','and','is','at','are','as','be','but','by','for','it','no','not','or'
        ,'such','that','their','there','these','to','was','with','they','will',  'v', 've', 'd']#, 's']
    # cleaned = re.sub('[\W_]+', ' ', text.encode('ascii', 'ignore').decode('ascii'))
    cleaned = re.sub('[\W_]+', ' ', str(text).encode('ascii', 'ignore').decode('ascii')).lower()
    # feature_one = re.sub(' +', '', cleaned).strip()
    feature_one = re.sub(' +', ' ', cleaned).strip()
    # feature_one = feature_one.replace(" s ", "''s  ")
    punct = [',', '.', '!', ';', ':', '?', "'", '"']
    for x in stopwords:
        feature_one = feature_one.replace(' {} '.format(x), ' ')
        if feature_one.startswith('{} '.format(x)):
            feature_one = feature_one[len('{} '.format(x)):]
        if feature_one.endswith(' {}'.format(x)):
            feature_one = feature_one[:-len(' {}'.format(x))]

    for x in punct:
        feature_one = feature_one.replace('{}'.format(x), ' ')
    return feature_one


def get_dataset(file_name: str, use_default_path: bool = True) -> pd.DataFrame:
    """Reads dataset from file.

    Parameters
    ----------
    file_name : str
        Dataset file name (csv).

    use_default_path : bool
        True if default dataset path should be used.

    Returns
    -------
    pd.DataFrame
        Dataset.
    """
    base_url = '../datasets/'
    if use_default_path:
        file = pd.read_csv(base_url+file_name+'.csv', sep=',')
    else:
        file = pd.read_csv(file_name+'.csv', sep=',')
    # file = file.replace("'", "''")
    file = file.apply(lambda x: x.astype(str).str.lower())
    return file

def get_dataset_with_path(file_name: str, includes_path: bool = True) -> None:
    """Reads dataset from file.

    Parameters
    ----------
    file_name : str
        Dataset file name (csv).

    includes_path : bool
        True if file_name includes path.

    Returns
    -------
    pd.DataFrame
        Dataset.
    """
    base_url = '../datasets/'
    if not includes_path:
        file = pd.read_csv(base_url + file_name, sep=',')
    else:
        file = pd.read_csv(file_name, sep=',')
    # file = file.replace("'", "''")
    file = file.apply(lambda x: x.astype(str).str.lower())
    return file

def huffman_encode(frequency: Dict) -> List[str]:
    """Encodes dictionary using Huffman Code.

    Parameters
    ----------
    frequency : Dict
        Input symbols and weights.

    Returns
    -------
    List[str]
        List of codes.
    """
    heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def generate_list_of_list_from_string(s: str, delimiter: str = ', ') -> None:
    """Splits string.

    Parameters
    ----------
    s : str
        Input string.

    delimiter : str
        String is split at delimiters.

    Returns
    -------
    List[str]
        List of substrings.
    """
    for i in np.arange(len(s)):
        s[i] = list(s[i].split(delimiter))

    return s
