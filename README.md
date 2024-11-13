# Implementation of Peekaboo

This is the python implementations of the attacks presented in :

"Just a Peekaboo: Passive Attacks Against DSSE via Snapshot Observations"

Run the python files with name starting with ``test_*.py`` to simulate the attack under different situations.

Run the files starting with ``draw_*.py`` to draw the results after performing the experimental code.

## About the dataset
The folder ``dataset`` contains the datasets. 

For Enron, the ``enron.pkl`` contains a dictionary-type object, where the different keys correspond to different information. 
1. The ``keywords_list`` maps to a list of keywords. We then use the corresponding index value to represent the keyword in code. 
2. The ``keywords_trend`` maps to a two-dimension-numpy.ndarray matrix, where ``keywords_trend[i][j]`` indicates the query trend of the keyword corresponding to index i on day j starting from July 2019. 
3. The ``keywords_count`` maps to a list, which stores the number of files that contain keywords. 
4. The ``total_doc`` maps to a list of files, which contain the index values for all keywords that appear in the file. 
5. The ``doc_size`` maps to a list that stores the number of bytes of the files. 
6. The ``doc_timestamp`` maps to a list that records when the files were added.

The same structure is used for Lucene.

## About the attacks
This file folder contains the main attack algorithm code.

``FMA.py``: a replication of the passive attack against DSSE, FMA,  proposed in "Leakage-Abuse Attacks Against Forward and Backward Private Searchable Symmetric Encryption".

``ihop.py``: a replication of IHOP proposed in "IHOP: Improved Statistical Query Recovery against Searchable Symmetric Encryption through Quadratic Optimization".

``infer_sp.py``: implementation of the first part of Peekaboo.

``real_sp.py``: implementation for getting a real SP to set up the benchmarks.

``jingsaw_extension.py``: implementation of the second part of Peekaboo based on Jigsaw. Jigsaw is proposed in "Query Recovery from Easy to Hard: Jigsaw Attack against SSE".

``sap_extension.py``: implementation of the second part of Peekaboo based on Sap. Sap is proposed in "Hiding the Access Pattern is Not Enough: Exploiting Search Pattern Leakage in Searchable Encryption".

## About the ``test_*.py`` and the other files 
``countermeasures.py``: implementations of countermeasures, file volume pattern padding and access pattern obfuscation.

``simulate_leakage.py``: simulates the client's behaviors and generates the leakage revealed for the attacker according to different parameters.

``run_deduce_sp.py``: takes the parameters of a deduce sp and simulates the deduce sp under the given condition.

``run_single_attack.py``: takes the parameters of an attack and simulates the attack under the given condition.

``test_accuracy_obfuscation.py``: tests Jigsaw, Jigsaw with SP, Sap, Sap with SP, and FMA against obfuscation.

``test_accuracy_padding_file_size.py``: tests Jigsaw, Jigsaw with SP, Sap, Sap with SP, and FMA against padding.

``test_accuracy_observed_query_number.py``: tests the effect of the number of observed queries on the attack accuracy.

``test_accuracy_offline_days.py``: tests the effect of the number of offline day on the attack accuracy.

``test_accuracy_rounds.py``: tests the effect of the number of round on the attack accuracy.

``test_ARI_days.py``: tests the effect of the number of observation days on the ARI.

``test_ARI_rounds.py``: tests the effect of the number of rounds on the ARI.

``utils.py``: calculates attack accuracy