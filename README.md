# Q-recsys (Query-based recommender system)

This library allows you to query the most transactionally similar item using semantic information based on a list of users, items and user-item interactions.

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [How it works](#how-it-works)
* [Key features of algorithm](#key-features-of-algorithm)
* [Use cases](#use-cases)

## Requirements

Python >= 3.4

```bash
pip install -r requirements.txt
```

It is recommended to install this library in a conda environment.

## Quick start

1. Prepare `users.csv`, `items.csv`, and `interactions.csv`
files. These CSV files require these formats:

    `users.csv`

    ```text
    id
    0
    1
    2
    ```

    `items.csv` (note that the `title` column must be present):

    ```text
    id,title
    0,machine learning
    1,financial markets
    2,sleep deprivation
    3,sustainable environment
    ```

    `interactions.csv` (note that the user and item are IDs defined in `users.csv` and `items.csv` respectively)

    ```text
    user,item,interaction
    1,0,1
    2,2,1
    0,3,1
    ```

2. Run the following:

    Import the relevant function and class.

    ```python
    >>> from qrecsys import preprocess, Recommender
    ```

    Preprocess the data and serialise the embeddings. By default, it looks for interactions.csv and items.csv in the current directory.

    ```python
    >>> preprocess(path_interactions="samples/interactions.csv",
                   path_items="samples/items.csv")
    ```

    Instantiate the recommender (it will look for the serialised data files). Then recommend items based on a query.

    ```python
    >>> recommender = Recommender(path_interactions="samples/interactions.csv",
                                  path_items="samples/items.csv")
    >>> recommender.recommend("politics")
    ['Contentious politics',
    'Globalisation, environment and social justice : perspectives, issues and concerns',
    'The will to improve : governmentality, development, and the practice of politics',
    'New state spaces : urban governance and the rescaling of statehood',
    'Shadows in the forest : Japan and the politics of timber in Southeast Asia']
    ```

## How it works

Every item is encoded as two vector representations: the *semantic embedding* and *transactional embedding*.

Semantic embeddings are found by encoding the title of every item using Universal Sentence Encoder (USE).

Transactional embeddings are the latent representations found by matrix factorisation, a common collaborative filtering technique.

Here is what happens during retrieval:

1. The query is semantically encoded using USE and we find the most similar items in the semantic embedding space.
2. We obtain the respective transactional embeddings of the items from above and return it to the user.

## Key features of algorithm

* **Transactional similarity** Similar items based on users reading history
* **Semantic similarity** Similar items based on semantics of title
* **Implicit feedback** User interactions do not explicitly indicate that the user 'liked' it, rather representing a transaction
