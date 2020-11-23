# Q-recsys (Query-based recommender system)

This library allows you to query the most transactionally similar item using semantic information based on a list of users, items and user-item interactions.

It is beneficial that users of this repo are somewhat familiar with these concepts:
collaborative filtering, deep learning and nearest neighbours.

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [How it works](#how-it-works)
* [Key features of algorithm](#key-features-of-algorithm)
* [Use cases](#use-cases)

## Requirements

* Python >= 3.4
* C++ compiler to install `implicit`. See [here](https://github.com/benfred/implicit#installation)
for more details.

We recommend installing the dependencies using conda:

```bash
conda create -n qrecsys python==3.7
pip install -r requirements.txt
```

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
    >>> preprocess(path_interactions="interactions.csv",
                   path_items="items.csv")
    ```

    Instantiate the recommender (it will look for the serialised data files). Then recommend items based on a query.

    ```python
    >>> recommender = Recommender(path_interactions="interactions.csv",
                                  path_items="items.csv")
    >>> recommender.recommend("politics")
    ['Contentious politics',
    'Globalisation, environment and social justice : perspectives, issues and concerns',
    'The will to improve : governmentality, development, and the practice of politics',
    'New state spaces : urban governance and the rescaling of statehood',
    'Shadows in the forest : Japan and the politics of timber in Southeast Asia']
    ```

## How it works

The whole process can be divided into 2 stages.

![qrecsys.png](qrecsys.png)

**Stage 1: Create title embeddings**

Every item is encoded as two vector representations: the *semantic embedding* and *transactional embedding*.
Semantic embeddings are found by encoding the title of every item using Universal Sentence Encoder (USE).
Transactional embeddings are the latent representations found by matrix factorisation (MF), a common collaborative filtering technique.

The `process` function in the `qrecsys` module takes care of reading the CSV files, encoding the titles into USE and MF embeddings,
then serialising these vector representations to be used in the next step.

**Step 2: Retrieval**

Here is what happens in the retrieval stage:

1. Read the serialised vector representations.
2. The query is semantically encoded using USE and we find the most similar items in the semantic embedding space.
3. We obtain the respective transactional embeddings of the items from above and return it to the user.

## Key features of algorithm

* **Transactional similarity** Similar items based on users reading history
* **Semantic similarity** Similar items based on semantics of title
* **Implicit feedback** User interactions do not explicitly indicate that the user 'liked' it, rather representing a transaction
