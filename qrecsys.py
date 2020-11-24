import math
from typing import List
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

EMBEDS_USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"


def preprocess(path_interactions: str = "interactions.csv",
               path_items: str = "items.csv",
               path_serialised: str = ".",
               embeds_use_url: str = EMBEDS_USE_URL,
               embeds_mf_dim: int = 8):
    """
    Read data, fit the data, convert title to USE & MF embeddings and
    serialise these embeddings.

    Args:
        path_interactions (str, optional): Path to interactions file.
            Defaults to "interactions.csv".
        path_items (str, optional): Path to items file.
            Defaults to "items.csv".
        path_serialised (str, optional): Directory to save the serialised data.
            Defaults to ".".
        embeds_use_url (str, optional): The URL of the USE encoder model to use
            from TF Hub. Defaults to EMBEDS_USE_URL (version 4).
        embeds_mf_dim (int, optional): The no. of dimensions of the MF embedding.
            Defaults to 8.

    Raises:
        FileNotFoundError: If "path_interactions" is invalid.
        FileNotFoundError: If "path_serialised" is invalid.
        ColumnNotFoundError: If "user", "item", and "interaction"
            columns are not found in the interactions file
        ColumnNotFoundError: If "title" column not found in the
            items file
    """
    # Check file paths
    path_interactions = Path(path_interactions)
    path_items = Path(path_items)
    path_serialised = Path(path_serialised)
    if not path_interactions.exists():
        raise FileNotFoundError("Specify a file for interactions")
    if not path_items.exists():
        raise FileNotFoundError("Specify a file for items")
    if not path_serialised.exists():
        path_serialised.mkdir()

    # Read data
    df_intxn = pd.read_csv(path_interactions)
    df_items = pd.read_csv(path_items, index_col="id")
    if set(df_intxn.columns) - set(["interaction", "item", "user"]):
        raise ColumnNotFoundError(
            "These columns must be present in interactions: "
            f"{str(['interaction', 'item', 'user'])}")
    if "title" not in df_items.columns:
        raise ColumnNotFoundError("`title` must be present in items")

    # Aggregate interactions data
    df_intxn = df_intxn.groupby(["user","item"]).sum().reset_index()

    # Format to usable data
    # Titles are batched to encoder the titles
    # CSR matrix is a sparse matrix that is used in implicit
    titles = df_items["title"].tolist()
    batched_titles = batch(titles)
    mat = csr_matrix(
        (df_intxn["interaction"], (df_intxn["item"], df_intxn["user"])))

    # USE model
    model_use = hub.load(embeds_use_url)

    # MF Model
    model_mf = implicit.als.AlternatingLeastSquares(factors=embeds_mf_dim)
    model_mf.fit(mat)

    # Title embeddings encoded using USE & MF respectively
    embeds_use = [model_use(batched).numpy() for batched in batched_titles]
    embeds_use = np.vstack(embeds_use)
    embeds_mf = model_mf.item_factors.copy()

    # Serialise embeddings
    np.save(path_serialised/"embeds_use.npy", embeds_use)
    np.save(path_serialised/"embeds_mf.npy", embeds_mf)


class Recommender:

    def __init__(self,
                 path_items: str = "items.csv",
                 path_interactions: str = "interactions.csv",
                 path_serialised: str = "."):

        path_items = Path(path_items)
        path_interactions = Path(path_interactions)
        path_embeds_mf = Path(path_serialised)/"embeds_mf.npy"
        path_embeds_use = Path(path_serialised)/"embeds_use.npy"

        self.encoder = hub.load(EMBEDS_USE_URL)
        self.embeds_mf = np.load(path_embeds_mf)
        self.embeds_use = np.load(path_embeds_use)
        self.items = pd.read_csv(path_items, index_col="id")
        self.interacted_items = set(pd.read_csv(path_interactions)["item"])

    def recommend(self,
                  query: str,
                  *,
                  K_use: int = 2,
                  K_mf: int = 5,
                  n_to_recommend: int = 5,
                  use_buffer_multiplier: int = 10,
                  mf_buffer_multiplier: int = 10) -> list:
        """Generate a list of recommendations based on a search query

        Args:
            query (str): Search query to provide context.
            K_use (int, optional): To retrieve top items based on semantic
                similarity. Defaults to 2.
            K_mf (int, optional): To retrieve top items based on transactional
                similarity. Defaults to 5.
            n_to_recommend (int, optional): The no. of recommendations that will
                be generated. Defaults to 5.
            use_buffer_multiplier (int, optional): this is needed because we will
                filter out items that have not been interacted. Defaults to 1000.
            mf_buffer_multiplier (int, optional): this is needed because we need to
                ensure the similar mf items don't already exist in the final
                recommendations. Defaults to 10.

        Returns:
            list: item recommendations as item IDs
        """
        # Get encoding
        encoded_query = self.encoder([query])
        encoded_query = encoded_query.numpy()

        # 1. Get nearest USE items
        item_ids = self._find_nearest(
            encoded_query,
            self.embeds_use,
            K=K_use*use_buffer_multiplier)

        # 2. Filter out to items that have not been interacted with
        item_ids = [item_id for item_id in item_ids
                    if item_id in self.interacted_items]

        # 3. Get top `K_use`
        item_ids = item_ids[:K_use]

        # 4. For every item, get `K_mf` neighbours in the MF space
        recs = []
        for item_id in item_ids:
            mf_items = self._find_nearest(
                self.embeds_mf[None, item_id],
                self.embeds_mf,
                K=K_mf*mf_buffer_multiplier)
            rec = np.setdiff1d(mf_items, recs, assume_unique=True)[:K_mf]
            recs.extend(rec)

        # 5. Truncate
        recs = recs[:n_to_recommend]

        # 6. Get titles
        recs_titles = [self.items.loc[idx].item() for idx in recs]

        return recs_titles


    def _find_nearest(self, x, y, K: int) -> list:
        """Find K nearest neighbours from a list of embeddings given a query.
        Similarity metric is calculated using Euclidean distance.

        Args:
            x (array-like): Query embedding
            y (array-like): A list of embeddings
            K (int): No. of neighbours to retrieve

        Returns:
            list: Nearest neighbours
        """
        dists = euclidean_distances(x, y)
        sorted_items = np.argsort(dists)[::-1]
        return sorted_items[0][:K]


def batch(items: List[str], batch_size=32) -> List[List[str]]:
    """Batch a list of strings to a list of lists having a maximum
    of 32 items.

    Args:
        items (List[str]): A list of items
        batch_size (int, optional): Defaults to 32.

    Returns:
        List[List[str]]: A list of batched items
    """
    num_batches = math.ceil(len(items) / batch_size)
    batched_items = [
        items[batch_size*batch_idx:batch_size*(batch_idx+1)]
        for batch_idx in range(num_batches)]
    return batched_items


class ColumnNotFoundError(Exception):
    pass
