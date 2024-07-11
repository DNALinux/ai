With BLAST Manual (PDF)
=======================


Using data with pre-processing (Embeddings: 3)
----------------------------------------------

Question 1: What is word size parameter in BLAST?

The word size parameter in BLAST refers to the number of amino acids or nucleotides used as a unit for comparison during a sequence search.

Question 2: How to get the results of BLASTP in XML format?

To get the results of BLASTP in XML format, use the -outfmt option followed by "5" or "xml". For example:

```
blastp -query query.fasta -db database -out blast.xml -outfmt 5
```

Question 3: How to perform a BLAST on a specific taxonomic group?

To perform a BLAST on a specific taxonomic group, use the "-taxid" option followed by the taxonomic ID of the group you are interested in. For example: `blastp -db nr -query query.fasta -max_target_seqs 1000 -evalue 1e-10 -outfmt 6 -taxid 2` This will search for sequences similar to those in your query file within the NR database, but only considering sequences from the bacterial kingdom (taxonomic ID 2).

Question 4: What blast program do I use to perform BLAST with epitopes smaller than 10 amino acids?

blastnshort or dcmegablast.

Question 5: Which kind of databases can be searched with BLASTX?

Protein databases.

**TIME: real    4m17.925s**


Using data with pre-processing (Embeddings: 5)
----------------------------------------------

Question 1: What is word size parameter in BLAST?

The word size parameter determines how many words (i.e., units of sequence) are considered during a BLAST search.

Question 2: How to get the results of BLASTP in XML format?

Use the command-line option `-outfmt 5` with the BLASTP program.

Question 3: How to perform a BLAST on a specific taxonomic group?

To perform a BLAST on a specific taxonomic group, you can use the "taxid" option in the BLAST command line application. For example:

`blastp -db nr -query query.fasta -out output.txt -taxid 9606`

This will search for similarities between the sequences in your query file and all organisms that belong to the taxonomic group with the given taxid (in this case, humans).

Question 4: What blast program do I use to perform BLAST with epitopes smaller than 10 amino acids?

blastp

Question 5: Which kind of databases can be searched with BLASTX?

Protein database.


**TIME real    4m36.454s**

Using data without pre-processing (Embeddings: 3)
-------------------------------------------------

Question 1: What is word size parameter in BLAST?

Length of initial exact match.

Question 2: How to get the results of BLASTP in XML format?

Use blast_formatter with the –outfmt 5 and –xml options. For example:

```
$blast_formatter –rid X3R7GAUS014 –out test.xml –outfmt 5
```

Question 3: How to perform a BLAST on a specific taxonomic group?

To perform a BLAST on a specific taxonomic group, use the -taxids option and provide the NCBI taxonomy ID(s) (taxid) for the given organism(s). For example, 9606 is the taxid for human.

Question 4: What blast program do I use to perform BLAST with epitopes smaller than 10 amino acids?

blastp-fast

Question 5: Which kind of databases can be searched with BLASTX?

Protein databases.

Using data without pre-processing (Embeddings: 5)
-------------------------------------------------

Question 1: What is word size parameter in BLAST?

There is no word size parameter mentioned in the provided data.

Question 2: How to get the results of BLASTP in XML format?

Use the option "outfmt 5".

Question 3: How to perform a BLAST on a specific taxonomic group?

Limiting a Search by taxonomy: Set virtual memory to unlimited, then run `blastp -db nr -query QUERY.fsa -outfmt “7 std taxid” -taxon your_taxon_id`.

Question 4: What blast program do I use to perform BLAST with epitopes smaller than 10 amino acids?

blastp

Question 5: Which kind of databases can be searched with BLASTX?

Protein databases.


**TIME real    3m20.357s**


