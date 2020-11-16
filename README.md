# Query-based recommeder system

This is a quasi search engine recommender system that takes in a query and returns transactionally similar items. This method is used in NUS Libraries.

* [Introduction](##introduction)
* [Requirements](##requirements)
* [Quick start](##quick-start)
* [How it works](##how-it-works)

## Requirements

Python >= 3.4

```bash
pip install -r requirements.txt
```

It is recommended to install this library in a conda environment.

## Quick start

1. Prepare `users.csv`, `items.csv`, and `interactions.csv`
files. These CSV files require these formats:

    `users.csv`:

    ```text
    id
    0
    1
    2
    ```

    `items.csv`:

    Note that the `title` column must be present.

    ```text
    id,title
    0,machine learning
    1,financial markets
    2,sleep deprivation
    3,sustainable environment
    ```

    `interactions.csv`:

    Note that the user and item are IDs defined in `users.csv` and `items.csv` respectively.

    ```text
    user,item,interaction
    1,0,1
    2,2,1
    0,3,1
    ```

2. Run the following:

    ```python
    >>> from recsys_pipeline import preprocess, Recommender
    >>> # looks for interactions.csv and items.csv in the current directory
    >>> preprocess()
    >>> recommender = Recommender()
    >>> recommender.recommend("sustainable environment")
    [3, 0, 1, 2]
    ```

## How it works

Every item is encoded as two vector representations: the *semantic embedding* and *transactional embedding*.

Semantic embeddings are found by encoding the title of every item using Universal Sentence Encoder (USE).

Transactional embeddings are the latent representations found by matrix factorisation, a common collaborative filtering technique.

Here is what happens during retrieval:

1. the query is semantically encoded using USE and we find the most similar items in the semantic embedding space
2. We obtain the respective transactional embeddings of the items from above and return it to the user.
