# MATE

## paper MATE

MATE leverages a novel inverted index to disocver n-ary joinable tables from a large corpus of tables to a given input dataset and a set of query columns.
In this project, ```generate_index()``` in ```index_generation.py``` is responsible to build the inverted index and ```MATE()``` is the MATE system that discovers the joinable tables from a corpus of tables.

Having the traditional inverted index defined in DataXFormer paper [1], the user should generate our novel index using the following function:

```generate_index()```

Then, the joinable tables can be discovered with the ```MATE()``` function in ```mate_table_extraction``` class.


