from base import *
from dbhandler import *
import pyhash
from tqdm import tqdm
import re
from simhash import Simhash
from bloom_filter import BloomFilter
from heapq import heapify, heappush, heappop
import hashlib
from collections import Counter
import math
import numpy as np
from typing import List, Dict, Any, Tuple


class MATETableExtraction:
    """MATE:

    Parameters
    ----------
    dataset_name : str
        Name of the main inverted index table in the database.

    dataset_name : str
        Name of the query dataset.

    dataset_path : str
        Path of the query dataset csv file.

    query_column_list : List[str]
        List of query columns.

    t_k : int
        Top-k tables to return.

    inverted_index_table : str
        Name of the inverted index table.

    ones : int
        Number of ones to use for the XASH.

    log_file_name : str
        Name of the logfile.

    min_join_ratio : int
        Minimum join ratio.

    is_min_join_ratio_absolute : bool
        True, if minimum join ratio is absolute.
    """
    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 query_column_list: List[str],
                 t_k: int,
                 inverted_index_table: str,
                 ones: int = 5,
                 log_file_name: str = '',
                 min_join_ratio: int = 0,
                 is_min_join_ratio_absolute: bool = True
                 ):
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
        self.dbh = DBHandler(self.main_inverted_index_table_name)
        self.number_of_ones = ones
        self.log_file_name = log_file_name
        self.input_size = len(self.input_data)
        self.min_join_ratio = min_join_ratio
        self.is_min_join_ratio_absolute = is_min_join_ratio_absolute
        self.original_data = self.input_data.copy()
        self.input_data = self.input_data[self.query_columns]

    def evaluate_rows(self, input_row: Any, col_dict: Dict) -> Tuple[bool, str]:
        """Evaluates a row.

        Parameters
        ----------
        input_row : Any
            Row to evaluate.

        col_dict : Dict
            Column dictionary.

        Returns
        -------
        Tuple[bool, str]
            bool: True, if matching columns were found.
            str: Matching column order, if matching columns were found.
        """
        values = list(col_dict.values())
        query_cols_arr = np.array(self.query_columns)
        query_degree = len(query_cols_arr)
        matching_column_order = ''
        for q in query_cols_arr[-(query_degree - 1):]:
            q_index = list(self.input_data.columns.values).index(q)
            if input_row[q_index] not in values:
                return False, ''
            else:
                for col_id, val in col_dict.items():
                    if val == input_row[q_index]:
                        matching_column_order += '_{}'.format(str(col_id))
        return True, matching_column_order

    def XASH(self, token: str, hash_size: int = 128) -> int:
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

        return int(result) | int(math.pow(2, len(token)%(hash_size-length_bit_start)) * math.pow(2, length_bit_start))


    @staticmethod
    def get_simhash_features(s: str) -> List[str]:
        """Returns SIM Hash features.

        Parameters
        ----------
        s : str
            Input value.

        Returns
        -------
        List[str]
            Features.
        """
        width = 3
        s = s.lower()
        s = re.sub(r'[^\w]+', '', s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

    def generate_SIM_hash(self, hash_dict: Dict, token: str, hash_size: int) -> Tuple[Dict, int]:
        """Calculates SIM Hash for token.

        Parameters
        ----------
        hash_dict : Dict
            Dictionary of already computed hash values.

        token : str
            Input token.

        hash_size : int
            Number of bits.

        Returns
        -------
        Tuple[Dict, int]
            Dict: Updated hash_dict.
            int: Hash for given token.
        """
        if token in hash_dict:
            return hash_dict, hash_dict[token]
        simh = Simhash(self.get_simhash_features(token), f=hash_size).value
        hash_dict[token] = simh
        return hash_dict, simh

    @staticmethod
    def generate_CITY_hash(hash_dict: Dict, token: str, hash_size: int) -> Tuple[Dict, int]:
        """Calculates CITY Hash for token.

        Parameters
        ----------
        hash_dict : Dict
            Dictionary of already computed hash values.

        token : str
            Input token.

        hash_size : int
            Number of bits.

        Returns
        -------
        Tuple[Dict, int]
            Dict: Updated hash_dict.
            int: Hash for given token.
        """
        if token in hash_dict:
            return hash_dict, hash_dict[token]
        if hash_size == 128:
            hasher = pyhash.city_128()
        elif hash_size == 256:
            hasher = pyhash.city_fingerprint_256()
        cityh = hasher(token)
        hash_dict[token] = cityh
        return hash_dict, cityh

    @staticmethod
    def generate_MURMUR_hash(hash_dict: Dict, token: str, hash_size: int) -> Tuple[Dict, int]:
        """Calculates MURMUR Hash for token.

        Parameters
        ----------
        hash_dict : Dict
            Dictionary of already computed hash values.

        token : str
            Input token.

        hash_size : int
            Number of bits.

        Returns
        -------
        Tuple[Dict, int]
            Dict: Updated hash_dict.
            int: Hash for given token.
        """
        if token in hash_dict:
            return hash_dict, hash_dict[token]
        if hash_size == 128:
            hasher = pyhash.murmur3_x64_128()
        murmurh = hasher(token)
        hash_dict[token] = murmurh
        return hash_dict, murmurh

    @staticmethod
    def generate_MD5_hash(hash_dict: Dict, token: str, hash_size: int) -> Tuple[Dict, int]:
        """Calculates MD5 Hash for token.

        Parameters
        ----------
        hash_dict : Dict
            Dictionary of already computed hash values.

        token : str
            Input token.

        hash_size : int
            Number of bits.

        Returns
        -------
        Tuple[Dict, int]
            Dict: Updated hash_dict.
            int: Hash for given token.
        """
        if token in hash_dict:
            return hash_dict, hash_dict[token]
        if hash_size == 128:
            hasher = hashlib.md5()
        hasher.update(token.encode('UTF-8'))
        md5h = int(hasher.hexdigest(), 16)
        hash_dict[token] = md5h
        return hash_dict, md5h

    def hash_row_vals(self, hash_function: Any, row: Any, hash_size: int) -> None:
        """Calculates Hash value for row.

        Parameters
        ----------
        hash_function : Any
            Hash function to use for hash calculation.

        row : Any
            Input row.

        hash_size : int
            Number of bits.

        Returns
        -------
        int
            Hash value for row.
        """
        hresult = 0
        for q in self.query_columns:
            d, hvalue = hash_function(row[q], hash_size)
            hresult = hresult | hvalue
        return hresult

    def ICS(self) -> int:
        """Runs ICS.

        Returns
        -------
        int
            Minimum number of unique values for query.
        """
        min_unique_value_number = 9999999999999
        best_query = ''
        for q in self.query_columns:
            if len(set(self.input_data[q])) < min_unique_value_number:
                best_query = q
                min_unique_value_number = len(set(self.input_data[q]))
        self.query_columns.insert(0, self.query_columns.pop(self.query_columns.index(best_query)))
        return min_unique_value_number

    def MATE(self, hash_size: int = 128) -> None:
        """Runs MATE using XASH.

        Parameters
        ----------
        hash_size : int
            Number of bits.
        """
        print('MATE')
        self.run_system(self.XASH)

    def SIMHASH(self, hash_size: int = 128) -> None:
        """Runs MATE using SIM Hash.

        Parameters
        ----------
        hash_size : int
            Number of bits.
        """
        print('Sim Hash')
        self.run_system(self.generate_SIM_hash, hash_size, False, True)

    def CITYHASH(self, hash_size: int = 128) -> None:
        """Runs MATE using CITY Hash.

        Parameters
        ----------
        hash_size : int
            Number of bits.
        """
        print('City Hash')
        self.run_system(self.generate_CITY_hash, hash_size, False, True)

    def MURMURHASH(self, hash_size: int = 128) -> None:
        """Runs MATE using MURMUR Hash.

        Parameters
        ----------
        hash_size : int
            Number of bits.
        """
        print('Murmur Hash')
        self.run_system(self.generate_MURMUR_hash, hash_size, False, True)

    def MD5(self, hash_size: int = 128) -> None:
        """Runs MATE using MD5 Hash.

        Parameters
        ----------
        hash_size : int
            Number of bits.
        """
        print('MD5 Hash')
        self.run_system(self.generate_MD5_hash, hash_size, False, True)

    def BF(self, hash_size: int = 128) -> None:
        """Runs MATE using BF Hash.

        Parameters
        ----------
        hash_size : int
            Number of bits.
        """
        print('BF Hash')
        self.run_system_bf(hash_size, False, True)

    def SCR(self) -> None:
        """Runs SCI.

        """
        print('Linear')
        self.run_system(False, True)

    def MCR(self, hash_size=128):
        """Runs SCI.

        """
        print('Multi SCI')
        self.run_system_multi_sci(False, True)

    def run_SCI_system(self, run_ics: bool = False, active_pruning: bool = True):
        """Runs SCI.

        Parameters
        ----------
        run_ics : bool
            True to run ICS.

        active_pruning : bool
            True to enable activate pruning.
        """
        print('{} DATASET'.format(self.dataset_name))
        row_block_size = 100
        total_match = 0
        total_approved = 0

        if run_ics:
            self.ICS()

        g = self.input_data.groupby([self.query_columns[0]])
        gd = {}
        for key, item in g:
            gd[key] = np.array(g.get_group(key))

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

                relevant_input_rows = gd[token]

                for input_row in relevant_input_rows:
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

    def run_system_multi_sci(self, run_ics: bool = False, active_pruning: bool = True):
        """Runs MCI.

        Parameters
        ----------
        run_ics : bool
            True to run ICS.

        active_pruning : bool
            True to enable activate pruning.
        """
        max_table_check = 500
        g = self.input_data.groupby([self.query_columns[0]])
        gd = {}
        for key, item in g:
            gd[key] = np.array(g.get_group(key))

        fetching_start_1 = time.time()
        table_row_1 = self.dbh.get_concatinated_posting_list(self.dataset_name, self.query_columns[0], self.input_data[self.query_columns[0]])
        fetching_time_1 = time.time() - fetching_start_1

        fetching_start_2 = time.time()
        table_row_2 = self.dbh.get_concatinated_posting_list(self.dataset_name, self.query_columns[1], self.input_data[self.query_columns[1]])
        fetching_time_2 = time.time() - fetching_start_2

        evaluation_time_start = time.time()
        joinability_dictionary = {}
        table_dictionary_1 = {}
        for i in table_row_1:
            if str(i) == 'None':
                continue
            tableid = int(i.split(';')[0].split('_')[0])
            if tableid in table_dictionary_1:
                table_dictionary_1[tableid] += [i]
            else:
                table_dictionary_1[tableid] = [i]

        table_dictionary_2 = {}
        for i in table_row_2:
            if str(i) == 'None':
                continue
            tableid = int(i.split(';')[0].split('_')[0])
            if tableid in table_dictionary_2:
                tablerowid_i = i.split(';')[0]
                colid_i = i.split(';')[1].split('$')[0].split('_')[0]
                token_i = i.split(';')[1].split('$')[0].split('_')[1]
                if tablerowid_i in table_dictionary_2[tableid]:
                    table_dictionary_2[tableid][tablerowid_i] += [[colid_i, token_i]]
                else:
                    table_dictionary_2[tableid][tablerowid_i] = [[colid_i, token_i]]
            else:
                tablerowid_i = i.split(';')[0]
                colid_i = i.split(';')[1].split('$')[0].split('_')[0]
                token_i = i.split(';')[1].split('$')[0].split('_')[1]
                table_dictionary_2[tableid] = {}
                table_dictionary_2[tableid][tablerowid_i] = [[colid_i, token_i]]

        for tableid in sorted(table_dictionary_1, key=lambda k: len(table_dictionary_1[k]), reverse=True)[:max_table_check]:
            for i in table_dictionary_1[tableid]:
                tablerowid_1 = i.split(';')[0]
                # rowid_1 = tablerowid_1.split('_')[1]
                colid_1 = i.split(';')[1].split('$')[0].split('_')[0]
                token_1 = i.split(';')[1].split('$')[0].split('_')[1]

                if tableid in table_dictionary_2 and tablerowid_1 in table_dictionary_2[tableid]:
                    row_occurrences = table_dictionary_2[tableid][tablerowid_1] #list of [[colid, token]]
                    for row_occurrence in row_occurrences:
                        colid_2 = row_occurrence[0]
                        token_2 = row_occurrence[1]
                        relevant_rows = gd[token_1]
                        for relevant_row in relevant_rows:
                            if token_2 == relevant_row[1]:
                                if '{}_{}_{}'.format(tableid, colid_1, colid_2) in joinability_dictionary:
                                    joinability_dictionary['{}_{}_{}'.format(tableid, colid_1, colid_2)] += 1
                                else:
                                    joinability_dictionary['{}_{}_{}'.format(tableid, colid_1, colid_2)] = 1

        top_joinables = sorted(joinability_dictionary, key=lambda k: joinability_dictionary[k], reverse=True)[:self.top_k]
        ll = []
        for x in top_joinables:
            ll += [joinability_dictionary[x]]
        if len(ll)>0:
            joinability_average = np.average(ll)
        else:
            joinability_average = 0

        evaluation_time = time.time() - evaluation_time_start

        print(top_joinables)
        
        info_to_store = 'fetching_time_1 = {}, fetching_time_2 = {}, evaluation_time = {}, joinability_average = {} , '.format(fetching_time_1, fetching_time_2, evaluation_time, joinability_average)
        lg = logger(self.dataset_name)
        lg.log(self.log_file_name, info_to_store, 'Dataset: {}'.format(self.dataset_name), self.log_file_name)

    def run_system(self,
                   hash_function: Any,
                   hash_size: int = 128,
                   run_ics: bool = False,
                   active_pruning: bool = True
                   ):
        """Runs table extraction.

        Parameters
        ----------
        hash_function : Any
            Hash function to use.

        hash_size : int
            Number of bits.

        run_ics : bool
            True to run ICS.

        active_pruning : bool
            True to enable activate pruning.
        """
        print('{} DATASET'.format(self.dataset_name))
        row_block_size = 100
        total_match = 0
        total_approved = 0

        if run_ics:
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

    def hash_row_vals_bf(self, row: Any, hash_size: int) -> str:
        """Calculates Hash value for row using Bloom Filter.

        Parameters
        ----------
        row : Any
            Input row.

        hash_size : int
            Number of bits.

        Returns
        -------
        int
            Hash value for row.
        """
        bf = BloomFilter(6, hash_size, self.number_of_ones)
        for q in self.query_columns:
            bf.add(row[q])

        string_output = ''
        for i in bf.bit_array:
            if i:
                string_output += '1'
            else:
                string_output += '0'
        return string_output

    def run_system_bf(self, hash_size: int = 128, run_ics: bool = False, active_pruning: bool = True) -> int:
        """Runs table extraction using Bloom Filter.

       Parameters
       ----------
       hash_size : int
           Number of bits.

       run_ics : bool
           True to run ICS.

       active_pruning : bool
           True to enable activate pruning.
       """

        if len(self.input_data) == 0:
            return 0

        print('{} DATASET'.format(self.dataset_name))
        row_block_size = 10
        total_match = 0
        total_approved = 0

        if run_ics:
            self.ICS()

        self.input_data['SuperKey'] = self.input_data.apply(
            lambda row: self.hash_row_vals_bf(row, hash_size), axis=1)
        g = self.input_data.groupby([self.query_columns[0]])
        gd = {}
        for key, item in g:
            gd[key] = np.array(g.get_group(key))
        super_key_index = list(self.input_data.columns.values).index('SuperKey')

        top_joinable_tables = []  # each item includes: Tableid, joinable_rows
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
                    if (int(input_row[super_key_index], 2) | int(superkey, 2)) == int(superkey, 2):
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


if __name__ == '__main__':
    top_k = 10
    one_bits = 5
    bits = 128
    MATETableExtraction('movie', '../datasets/movie.csv', ['director_name', 'movie_title'], top_k, 'main_tokenized',
                        one_bits, f'MATE_datasets_k_bits_ones_{top_k}_{bits}_{one_bits}').MATE(bits)
