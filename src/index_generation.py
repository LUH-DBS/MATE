import vertica_python
import numpy as np
import math
from collections import Counter


def XASH(token: str, hash_size: int = 128) -> int:
    """Computes XASH for given token.

    Parameters
    ----------
    token : str
        Token.

    hash_size : int
        Number of bits.

    Returns
    -------
    int
        XASH value.
    """
    number_of_ones = 5
    char = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
    segment_size = segment_size_dict[hash_size]
    length_bit_start = 37 * segment_size
    result = 0
    cnt_dict = Counter(token)
    selected_chars = [y[0] for y in sorted(cnt_dict.items(), key=lambda x: (x[1], x[0]), reverse=False)[:number_of_ones]]
    for c in selected_chars:
        if c not in char:
            continue
        indices = [i for i, ltr in enumerate(token) if ltr == c]
        mean_index = np.mean(indices)
        token_size = len(token)
        for i in np.arange(segment_size):
            if mean_index <= ((i + 1) * token_size / segment_size):
                location = char.index(c) * segment_size + i
                break
        result = result | int(math.pow(2, location))

    # rotation
    n = int(result)
    d = int((length_bit_start * (len(token) % (hash_size - length_bit_start))) / (
                hash_size - length_bit_start))
    int_bits = int(length_bit_start)
    x = n << d
    y = n >> (int_bits - d)
    r = int(math.pow(2, int_bits))
    result = int((x | y) % r)

    result = int(result) | int(math.pow(2, len(token) % (hash_size - length_bit_start)) * math.pow(2, length_bit_start))

    return result

def generate_index(main_table: str = 'main_tokenized',
                   super_key_column: str = 'super_key',
                   hash_size: int = 128
    ) -> None:
    """Generates MATE index and stores it in DB.

    Parameters
    ----------
    main_table : str
        Main inverted index table.
        
    super_key_column : str

    hash_size : int
        Number of bits.
    """
    conn_info = {
        'host': 'SERVER_IP_ADDRESS',
        'port': 5433,
         'user': 'USERNAME',
         'password': 'PASSWORD',
         'database': 'DATABASE_NAME',
         'session_label': 'some_label',
         'read_timeout': 6000,
         'unicode_error': 'strict',
    }

    connection = vertica_python.connect(**conn_info)
    cur = connection.cursor()
    cur.execute(f'SELECT tableid, MAX(rowid) FROM {main_table} GROUP BY tableid;')

    for row in cur.fetchall():
        table_id = int(row[0])
        rowid_max = int(row[1])
        for rowid in np.arange(rowid_max):
            cur.execute(f'SELECT tokenized FROM {main_table} WHERE tableid = {table_id} AND rowid = {rowid};')

            row_tokens = cur.fetchall()
            row_tokens = [item for sublist in row_tokens for item in sublist]

            super_key = 0
            for token in row_tokens:
                super_key = super_key | XASH(str(token), hash_size)

            cur.execute(f'UPDATE main_tokenized SET {super_key_column} = {super_key} '
                        f'WHERE tableid = {table_id} AND rowid = {rowid}; '
                        f'COMMIT;')

generate_index()
