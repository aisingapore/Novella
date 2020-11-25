# Q-recsys (Query-based recommender system)

This library allows you to query the most transactionally similar item using semantic information based on a list of users, items and user-item interactions.

It is beneficial that users of this repo are somewhat familiar with these concepts:
collaborative filtering, deep learning and nearest neighbours.

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [How it works](#how-it-works)
* [Contributing](#contributing)

## Requirements

* Python >= 3.4
* C++ compiler is needed to install `implicit`. See [here](https://github.com/benfred/implicit#installation)
for more details. Note that if you have OpenBLAS or MKL installed, it is recommended to set the number of threads to 0 for optimal performance, as mentioned [here](https://github.com/benfred/implicit#optimal-configuration).

We recommend installing the dependencies using conda:

```bash
conda create -n qrecsys python==3.7
pip install -r requirements.txt
```


## Quick start

1. Prepare `users.csv`, `items.csv`, and `interactions.csv` files (alternatively, you can just make use of the sample files
    under `samples/` and skip this section). These CSV files require these formats:

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

    `interactions.csv` (note that the user and item are IDs defined in `users.csv` and `items.csv` respectively):

    ```text
    user,item,interaction
    1,0,1
    1,0,1
    2,2,2
    0,3,1
    ```

    Every line must be an instance of an interaction. For the above example, we have 4 transactions:
    * user `1` interacted with item `0` once
    * user `1` interacted with item `0` once
    * user `2` interacted with item `2` twice
    * user `0` interacted with item `3` once

    Note that it is fine to repeat transactions (like the first 2 lines) as they will be aggregated.

2. Run the following:

    Import the relevant function and class.

    ```python
    >>> from qrecsys import preprocess, Recommender
    ```

    Preprocess the data and serialise the embeddings.

    ```python
    >>> preprocess(path_interactions="interactions.csv",  # or samples/interactions.csv
                   path_items="items.csv")  # or samples/items.csv
    ```

    Instantiate the recommender (it will look for the serialised data files). Then recommend items based on a query.

    ```python
    >>> recommender = Recommender(path_interactions="interactions.csv",   # or samples/interactions.csv
                                  path_items="items.csv")  # or samples/items.csv
    >>> recommender.recommend("latent heat")  # if you are using our samples
    ['Convective heat transfer',
    'Procedures for simulating the performance of components and systems for energy calculations',
    'Roman power : a thousand years of empire',
    'The last generation of the Roman Republic',
    'Satoyama--satoumi ecosystems and human well-being : socio-ecological production landscapes of Japan']
    ```

## How it works

The recommender system has the following features:

* **Implicit feedback** User interactions do not explicitly indicate that the user 'liked' it, rather representing a transaction
* **Semantic similarity** Similar items can be found based on semantics of title
* **Transactional similarity** Similar items can be found based on other users' transactions with them

The whole process can be divided into 2 stages.

![qrecsys.png](qrecsys.png)

**Stage 0: Semantic and transactional title embeddings**

The first stage is attributed to `qrecsys.process` function.

Every item is first encoded as two vector representations: the *semantic embedding* and *transactional embedding*. Then, these representations are
serialised for use in the next stage.

Semantic embeddings are found by encoding the title of every item using Universal Sentence Encoder (USE). We use the USE encoder from [TF Hub](https://tfhub.dev) (this will be downloaded when you call `process`), trained on various data sources. The size of this embedding is 512. For example, here is the embedding for the title `Problem solving in analytical chemistry` (first item in `samples/users.csv`) truncated to the first 10 dims:

```
array([[ 6.64e-02, -7.70e-02,  2.66e-02,  2.28e-02,  6.38e-03, -6.71e-02,
        -5.27e-02, -5.45e-02,  8.83e-03,  5.14e-02, -5.92e-02, -1.17e-02,
       ...
       ], dtype=float32)
```

Transactional embeddings are the latent representations found by matrix factorisation (MF), a common collaborative filtering technique. We first format the interactions data into a sparse matrix then fit it using a weighted Alternated Least Squares optimiser, giving us a latent representation for every item. The size of this representation can be set in the `qrecsys.process` as the `embeds_mf_dim`. If you have a large number of items (>1M), we recommend setting the size of embedding to a higher number (eg. 256 or 512). Here is an example of an MF embedding for the title `Problem solving in analytical chemistry`:

```
array([ 1.17e-04,  3.14e-04, -5.00e-05, -1.21e-04,  1.19e-04, -1.43e-04,
       -5.16e-05,  3.34e-04], dtype=float32)
```

Note that an MF embedding will be 0's if no user has interacted with it.

**Step 1: Querying**

This stage is attributed to `qrecys.Recommender`. Here is what happens in the querying stage:

1. Read the serialised vector representations. This is done when a new instance of `Recommender` is created.
2. In `Recommender.recommend`, the query is first semantically encoded using USE. Then we find the `K_use` most similar items in the semantic embedding space.
3. For ever USE vector, we fetch `K_mf` most similar items in the MF embedding space.
4. Finally, we return `n_to_recommend` items to the user.

Note that there will be cases where similar items found in the USE space are mapped to items in the MF space that have not been interacted before (these vectors are 0). In this cases, we set `use_buffer_multiplier` and `mf_buffer_multiplier` can be increased accordingly so that we avoid this problem.

## Contributing

See any problems? Submit an issue and/or a PR 🤗!
