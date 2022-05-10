from typing import List
import pandas as pd
import vertica_python
from base import *
import os.path


class DBHandler:
    """Bloom filter using murmur3 hash function.

    Parameters
    ----------
    main_table_name : str
        Name of the main inverted index table in the database.
    """
    def __init__(self, main_table_name: str = 'main_tokenized'):
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
        self.cur = connection.cursor()
        self.main_table = main_table_name

    def get_concatinated_posting_list(self,
                                      dataset_name: str,
                                      query_column_name: str,
                                      value_list: pd.Series,
                                      top_k: int = -1,
                                      database_request: bool = True) -> List[str]:
        """Fetches posting lists for top-k values.

        Parameters
        ----------
        dataset_name : str
            Name of the query dataset.

        query_column_name: str
            Name of the query column.

        value_list : pd.Series
            Values to fetch posting lists for.

        top_k : int
            Number of posting lists to fetch. -1 to fetch all.

        database_request : bool
            If true, posting lists are fetched from the database. Otherwise cached files are used (if existing).

        Returns
        -------
        List[str]
            Posting lists for values.
        """
        if os.path.isfile("../cache/{}_{}_concatenated_posting_list.txt".format(dataset_name, query_column_name))\
                and top_k == -1 and not database_request:
            pl = []
            with open("../cache/{}_{}_concatenated_posting_list.txt".format(dataset_name, query_column_name), "r") as f:
                for line in f:
                    pl += [line.strip()]
            return pl
        else:
            distinct_clean_values = value_list.unique()
            joint_distinct_values = '\',\''.join(distinct_clean_values)
            if top_k != -1:
                query = f'SELECT DISTINCT concat(concat(concat(concat(concat(concat(' \
                        f'concat(concat(tableid,\'_\'), rowid), \';\'), colid), \'_\'), tokenized), \'$\'), superkey) ' \
                        f'FROM {self.main_table} ' \
                        f'WHERE REGEXP_REPLACE(REGEXP_REPLACE(tokenized, \'\W+\', \' \'), \' +\', \' \') ' \
                        f'  IN (\'{joint_distinct_values}\') LIMIT {top_k};'
            else:
                query = f'SELECT DISTINCT concat(concat(concat(concat(concat(concat(' \
                        f'concat(concat(tableid,\'_\'), rowid), \';\'), colid), \'_\'), tokenized), \'$\'), superkey) ' \
                        f'FROM {self.main_table} ' \
                        f'WHERE tokenized IN (\'{joint_distinct_values}\');'

            self.cur.execute(query)
            pl = [item for sublist in self.cur.fetchall() for item in sublist]

            if top_k == -1 and not database_request:
                with open("../cache/{}_{}_concatenated_posting_list.txt".format(dataset_name, query_column_name), "w") as f:
                    for s in pl:
                        f.write(str(s) + "\n")
            return pl

    def get_pl_by_table_and_rows(self, joint_list: List[str]) -> List[List[str]]:
        """Fetches posting lists a set of table_row_ids.

        Parameters
        ----------
        joint_list : List[str]
            List of table_row_ids.

        Returns
        -------
        List[List[str]]
            Posting lists for given table_row_ids.
        """
        distinct_clean_values = list(set(joint_list))
        joint_distinct_values = '\',\''.join(distinct_clean_values)
        tables = '\',\''.join(list(set([x.split('_')[0] for x in joint_list])))
        rows = '\',\''.join(list(set([x.split('_')[1] for x in joint_list])))
        query = f'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized ' \
                f'FROM {self.main_table} ' \
                f'WHERE tableid IN (\'{tables}\') ' \
                f'AND rowid IN(\'{rows}\') ' \
                f'AND concat(concat(tableid, \'_\'), rowid) IN (\'{joint_distinct_values}\');'
        self.cur.execute(query)
        pl = self.cur.fetchall()

        return pl
