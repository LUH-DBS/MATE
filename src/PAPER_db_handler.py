import vertica_python
import numpy as np
import pyhash
import pandas as pd
import random
from base import *
import math
from collections import Counter
from tqdm import tqdm
import os.path
import pickle
from io import StringIO


class db_handler:
    def __init__(self, main_table_name = 'main_tokenized', term_frequency_table='MATE_TF'):
        conn_info = {'host': 'SERVER_IP_ADDRESS',
                     'port': 5433,
                     'user': 'USERNAME',
                     'password': 'PASSWORD',
                     'database': 'DATABASE_NAME',
                     'session_label': 'some_label',
                     'read_timeout': 6000,
                     'unicode_error': 'strict',
                     }
        connection = vertica_python.connect(**conn_info)
        self.cur = connection.cursor()
        self.main_table = main_table_name

    def get_concatinated_posting_list(self, datasetname, query_column_name, value_list, topk=-1, DB_request = True):
        if os.path.isfile("../cache/{}_{}_concatenated_posting_list.txt".format(datasetname, query_column_name)) and topk == -1 and not DB_request:
            pl = []
            with open("../cache/{}_{}_concatenated_posting_list.txt".format(datasetname, query_column_name), "r") as f:
                for line in f:
                    pl += [line.strip()]
            return pl
        else:
            distinct_clean_values = value_list.unique()
            joint_distinct_values = '\',\''.join(distinct_clean_values)
            if topk != -1:
                query = 'SELECT distinct concat(concat(concat(concat(concat(concat(concat(concat(tableid,\'_\'), rowid), \';\'), colid), \'_\'), tokenized), \'$\'), superkey) from {} WHERE REGEXP_REPLACE(' \
                    'REGEXP_REPLACE(tokenized, \'\W+\', \' \'), \' +\', \' \') IN (\'{}\') LIMIT {};'.format(self.main_table, joint_distinct_values, topk)
            else:
                query = 'SELECT distinct concat(concat(concat(concat(concat(concat(concat(concat(tableid,\'_\'), rowid), \';\'), colid), \'_\'), tokenized), \'$\'), superkey) from {} WHERE ' \
                    'tokenized IN (\'{}\');'.format(self.main_table, joint_distinct_values)
            self.cur.execute(query)
            pl = [item for sublist in self.cur.fetchall() for item in sublist]

            if topk == -1 and not DB_request:
                with open("../cache/{}_{}_concatenated_posting_list.txt".format(datasetname, query_column_name), "w") as f:
                    for s in pl:
                        f.write(str(s) + "\n")
            return pl

    def get_pl_by_table_and_rows(self, joint_list):
        distinct_clean_values = list(set(joint_list))
        joint_distinct_values = '\',\''.join(distinct_clean_values)
        tables = '\',\''.join(list(set([x.split('_')[0] for x in joint_list])))
        rows = '\',\''.join(list(set([x.split('_')[1] for x in joint_list])))
        query = 'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized FROM {} WHERE tableid IN (\'{}\') AND rowid IN(\'{}\') AND concat(concat(tableid, \'_\'), rowid) IN (\'{}\');'.format(self.main_table, tables, rows, joint_distinct_values)
        self.cur.execute(query)
        pl = self.cur.fetchall()

        return pl
