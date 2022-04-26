# MATE

## Abstract

A core operation in data discovery is to find joinable tables for a given table. Real-world tables include both unary and n-ary join keys. However, existing table discovery systems are optimized for unary joins and are ineffective and slow in the existence of n-ary keys. In this paper, we introduce MATE, a table discovery system that leverages a novel hash-based index that enables n-ary join discovery through a space-efficient super key. We design a filtering layer that uses a novel hash, XASH. This hash function encodes the syntactic features of all column values and aggregates them into a super key, which allows the system to efficiently prune tables with non-joinable rows. Our join discovery system is able to prune up to 1000x more false positives and leads to over 60x faster table discovery in comparison to state-of-the-art.

## paper MATE

MATE leverages a novel inverted index to disocver n-ary joinable tables from a large corpus of tables to a given input dataset and a set of query columns.
In this project, ```generate_index()``` in ```index_generation.py``` is responsible to build the inverted index and ```MATE()``` is the MATE system that discovers the joinable tables from a corpus of tables.

Having the traditional inverted index defined in the DataXFormer paper, the user should generate our novel index using the following function:

```generate_index()```

Then, the joinable tables can be discovered with the ```MATE()``` function in ```mate_table_extraction``` class.


