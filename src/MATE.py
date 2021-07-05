import pandas as pd
from database_handler import *
import operator as op
from functools import reduce
import itertools
import time
from log_writer import *
import pyhash
from tqdm import tqdm
from multiprocessing import Pool, process
import random
import heapq
import sys
import re
from simhash import Simhash
from bloom_filter import BloomFilter
from heapq import heapify, heappush, heappop
from io import StringIO
# from unidecode import unidecode
import hashlib


class mate_table_extraction:
    def __init__(self, dataset_name, dataset_path, query_column_list, t_k, inverted_index_table, ones=5, log_file_name='', min_join_ratio=0,
                 is_min_join_ratio_absolute=True):
        self.input_data = pd.read_csv(dataset_path)
        self.main_inverted_index_table_name = inverted_index_table
        self.input_data = self.input_data.drop_duplicates(subset=query_column_list)
        for q in query_column_list:
            self.input_data[q] = self.input_data[q].apply(lambda x: get_cleaned_text(x)).replace('', np.nan).replace(
                'nan', np.nan).replace('unknown', np.nan)
            self.input_data.dropna(subset=[q], inplace=True)
        self.dataset_path = dataset_path
        self.query_columns = query_column_list
        self.top_k = t_k
        self.dataset_name = dataset_name
        self.dbh = db_handler(self.main_inverted_index_table_name)
        self.number_of_ones = ones
        self.log_file_name = log_file_name
        self.input_size = len(self.input_data)
        self.min_join_ratio = min_join_ratio
        self.is_min_join_ratio_absolute = is_min_join_ratio_absolute
        self.original_data = self.input_data.copy()
        self.input_data = self.input_data[self.query_columns]

    def XHash(self, token, hash_size=128):
        char = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
        segment_size = segment_size_dict[hash_size]
        length_bit_start = 37 * segment_size
        result = 0
        cnt_dict = Counter(token)
        selected_chars = [y[0] for y in
                          sorted(cnt_dict.items(), key=lambda x: (x[1], x[0]), reverse=False)[:self.number_of_ones]]
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

        n = int(result)
        d = int((length_bit_start * (len(token)%(hash_size-length_bit_start)))/(hash_size-length_bit_start))
        INT_BITS = int(length_bit_start)
        x = n << d
        y = n >> (INT_BITS - d)
        r = int(math.pow(2, INT_BITS))
        result = int((x | y) % r)

        result = int(result) | int(math.pow(2, len(token)%(hash_size-length_bit_start)) * math.pow(2, length_bit_start))

        return result

    def hash_row_vals(self, hashfunction, row, hash_size):
        hresult = 0
        for q in self.query_columns:
            d, hvalue = hashfunction(row[q], hash_size)
            hresult = hresult | hvalue
        return hresult

    def ICS(self):
        min_unique_value_number = 9999999999999
        best_query = ''
        for q in self.query_columns:
            if len(set(self.input_data[q])) < min_unique_value_number:
                best_query = q
                min_unique_value_number = len(set(self.input_data[q]))
        self.query_columns.insert(0, self.query_columns.pop(self.query_columns.index(best_query)))
        return min_unique_value_number

    def MATE(self, hash_size=128):
        print('MATE')
        self.run_system(self.XHash,
                        'superkey_{}'.format(hash_size), hash_size, False, True)

    def run_system(self, hash_function, hash_column_string_name, hash_size=128, run_ICS=False,
                   active_pruning=True):
        print('{} DATASET'.format(self.dataset_name))
        row_block_size = 100
        total_match = 0
        total_approved = 0

        if run_ICS:
            self.ICS()

        self.input_data['SuperKey'] = self.input_data.apply(
            lambda row: self.hash_row_vals(hash_function, row, hash_size), axis=1)
        g = self.input_data.groupby([self.query_columns[0]])
        gd = {}
        for key, item in g:
            gd[key] = np.array(g.get_group(key))
        super_key_index = list(self.input_data.columns.values).index('SuperKey')

        top_joinable_tables = []
        heapify(top_joinable_tables)

        table_row = self.dbh.get_concatinated_posting_list(self.dataset_name, self.query_columns[0],
                                                               self.input_data[self.query_columns[0]])
        table_dictionary = {}
        for i in table_row:
            if str(i) == 'None':
                continue
            tableid = int(i.split(';')[0].split('_')[0])
            if tableid in table_dictionary:
                table_dictionary[tableid] += [i]
            else:
                table_dictionary[tableid] = [i]
        candidate_external_row_ids = []
        candidate_external_col_ids = []
        candidate_input_rows = []
        candidate_table_rows = []
        candidate_table_ids = []

        overlaps_dict = {}
        pruned = False
        for tableid in tqdm(sorted(table_dictionary, key=lambda k: len(table_dictionary[k]), reverse=True)):
            set_of_rowids = set()
            hitting_posting_list_concatinated = table_dictionary[tableid]
            if active_pruning and len(top_joinable_tables) >= self.top_k and top_joinable_tables[0][0] >= len(
                    hitting_posting_list_concatinated):
                pruned = True
            if active_pruning and (
                    (self.is_min_join_ratio_absolute and len(hitting_posting_list_concatinated) < self.min_join_ratio)
                    or (not self.is_min_join_ratio_absolute and len(hitting_posting_list_concatinated) < round(
                    self.min_join_ratio * self.input_size))):
                pruned = True

            already_checked_hits = 0
            for hit in sorted(hitting_posting_list_concatinated):
                if active_pruning and len(top_joinable_tables) >= self.top_k and (
                        (len(hitting_posting_list_concatinated) - already_checked_hits + len(set_of_rowids)) <
                        top_joinable_tables[0][0]):
                    break
                tablerowid = hit.split(';')[0]
                rowid = tablerowid.split('_')[1]
                colid = hit.split(';')[1].split('$')[0].split('_')[0]
                token = hit.split(';')[1].split('$')[0].split('_')[1]
                superkey = int(hit.split('$')[1])

                relevant_input_rows = gd[token]

                for input_row in relevant_input_rows:
                    if (input_row[super_key_index] | superkey) == superkey:
                        candidate_external_row_ids += [rowid]
                        set_of_rowids.add(rowid)
                        candidate_external_col_ids += [colid]
                        candidate_input_rows += [input_row]
                        candidate_table_ids += [tableid]
                        candidate_table_rows += ['{}_{}'.format(tableid, rowid)]

                already_checked_hits += 1
            if pruned or len(candidate_external_row_ids) >= row_block_size:
                if len(candidate_external_row_ids) == 0:
                    break
                candidate_input_rows = np.array(candidate_input_rows)
                candidate_table_ids = np.array(candidate_table_ids)
                pls = self.dbh.get_pl_by_table_and_rows(candidate_table_rows)

                table_row_dict = {}
                for i in pls:
                    if i[0] not in table_row_dict:
                        table_row_dict[str(i[0])] = {}
                        table_row_dict[str(i[0])][str(i[1])] = str(i[2])
                    else:
                        table_row_dict[str(i[0])][str(i[1])] = str(i[2])
                for i in np.arange(len(candidate_table_rows)):
                    col_dict = table_row_dict[candidate_table_rows[i]]
                    match, matched_columns = self.evaluate_rows(candidate_input_rows[i], col_dict)
                    total_approved += 1
                    if match:
                        total_match += 1
                        complete_matched_columns = '{}{}'.format(str(candidate_external_col_ids[i]), matched_columns)
                        if candidate_table_ids[i] not in overlaps_dict:
                            overlaps_dict[candidate_table_ids[i]] = {}

                        if complete_matched_columns in overlaps_dict[candidate_table_ids[i]]:
                            overlaps_dict[candidate_table_ids[i]][complete_matched_columns] += 1
                        else:
                            overlaps_dict[candidate_table_ids[i]][complete_matched_columns] = 1
                for tbl in set(candidate_table_ids):
                    if tbl in overlaps_dict and len(overlaps_dict[tbl]) > 0:
                        join_keys = max(overlaps_dict[tbl], key=overlaps_dict[tbl].get)
                        joinability_score = overlaps_dict[tbl][join_keys]
                        if self.top_k <= len(top_joinable_tables):
                            if top_joinable_tables[0][0] < joinability_score:
                                popped_table = heappop(top_joinable_tables)
                                heappush(top_joinable_tables, [joinability_score, tbl, join_keys])
                        else:
                            heappush(top_joinable_tables, [joinability_score, tbl, join_keys])
                candidate_external_row_ids = []
                candidate_external_col_ids = []
                candidate_input_rows = []
                candidate_table_rows = []
                candidate_table_ids = []

                overlaps_dict = {}

            if pruned:
                break

        print('---------------------------------------------')
        print(top_joinable_tables)
        print(len(top_joinable_tables))
        print('FP = {}'.format(total_approved - total_match))


top_k = 10
one_bits = 5
bits = 128
mate_table_extraction('movie', '../datasets/movie.csv', ['director_name', 'movie_title'], top_k, 'main_tokenized', one_bits, 'MATE_datasets_k_bits_ones_{}_{}_{}'.format(top_k, bits, one_bits)).MATE(bits, True)
