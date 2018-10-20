# Co-oc_embedding

This project was inspired by the GloVe[9] model used in Natural Language Processing(NLP). GloVe trains word embeddings using a word co-occurrence matrix. The success of such co-occurrence em- bedding in NLP field motivates us to explore its power in some other tasks.<br>
The specific goal of this project is to identify which industry does an entity belong to using its embedding vector trained from a co-occurrence matrix, such an algorithm would help to improve Named Entity Recognition ideally. The designed pipeline would start from extracting entities from news articles with Named Entity Recognition (NER), building an entity co-occurrence matrix, training entity embeddings with GloVe and then train classifiers with embedding vectors.<br>
During the project, we have also found a lot of valuable knowledge about co-occurrence embedding and some of them could be used beyond this project, which we will talk more about in the next few sections.
