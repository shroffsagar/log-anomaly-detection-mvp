from typing import Any, Dict, Optional
import numpy as np
from numpy import random
import joblib
from rrcf import RCTree


class RCFDetector:
    """Thin adapter over RRCF with a stable API: fit / score / save / load."""

    def __init__(self):
        self.forest: list[RCTree] | None = None
        self.params: Dict[str, Any] = {}
        self.n_features_: Optional[int] =  None
        self.is_fitted: bool = False

    def _validate(self, X: np.ndarray):
        """
            Validates the X for the shape and samples provided
        """
        if(X.ndim != 2):
            raise ValueError(f"X.shape is not [n_samples, n_features]")
        n_samples, n_features = X.shape
        if(n_samples == 0):
            raise ValueError(f"No samples provided in X")
    
    def fit(self, X: np.ndarray, **params):
        """
        Trains the RRCF detector on feature matrix X (shape: [n_samples, n_features])
        Params:
            n_trees: int = 40
            tree_size: int = 256
            random_state: Optional[int] = None
        """
        # Validation
        self._validate(X)
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        # params & defaults
        n_trees = int(params.get("n_trees", 40))
        tree_size = int(params.get("tree_size", 256))
        random_state: Optional[int] = params.get("random_state", None)
        rng = np.random.default_rng(seed=random_state)
        self.params = {
            "n_trees" : n_trees,
            "tree_size" : tree_size,
            "random_state" : random_state
        }
        self.n_features_ = n_features
        # build forest
        forest = []
        for _ in range(n_trees):
            replace = n_samples < tree_size
            idx = rng.choice(n_samples, tree_size, replace=replace)
            tree = RCTree()
            # for every sub-sample row
            for j in idx: 
                j = int(j)
                tree.insert_point(X[j], j)
            forest.append(tree)
        self.forest = forest
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores (higher = more anomalous) using avg codisp 
        across trees by temp add/removing. X must be of shape (n_samples, n_features)
        """
        # Validate before scoring anomalies
        if not self.is_fitted or self.forest is None:
            raise RuntimeError("Call fit(x,...) before score()")
        self._validate(X)
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        if(n_features != self.n_features_):
            raise ValueError(f"Expected shape (_, {self.n_features_}) but got {X.shape}")
        # calculate the score for every sample
        scores = np.empty(n_samples, dtype=float)
        for i, x in enumerate(X):
            temp_indx = ("score", i)
            s = 0
            for tree in self.forest:
                tree.insert_point(x, temp_indx)
                s += tree.codisp(temp_indx)
                tree.forget_point(temp_indx)
            avg_s = s / len(self.forest)
            scores[i] = avg_s
        return scores

    def save(self, path: str):
        """Persist the trained detector to `path`"""
        if not self.is_fitted or self.forest is None:
            raise RuntimeError("Cannot save: model is not fitted.")
        # Convert each tree to a plain Python dict (safe to pickle or JSON)
        forest_dicts = []
        for tree in self.forest:
            obj = tree.to_dict()  # official API
            forest_dicts.append(obj)

        payload = {
            "meta": {"class": self.__class__.__name__, "version": "1"},
            "state": {
                "params": self.params,
                "n_features_": self.n_features_,
                "forest": forest_dicts,   # list of nested dicts
            },
        }
        import joblib
        joblib.dump(payload, path, compress=3)


    @classmethod
    def load(cls, path: str) -> "RCFDetector":
        """Load a persisted detector from `path` by reconstructing trees."""
        import joblib, rrcf, numpy as np
        payload = joblib.load(path)
        state = payload.get("state", payload)
        obj = cls()
        obj.params = state["params"]
        obj.n_features_ = state["n_features_"]
        # Rebuild each RCTree from its dict
        forest = []
        for obj_dict in state["forest"]:
            tree = rrcf.RCTree.from_dict(obj_dict)
            forest.append(tree)
        obj.forest = forest
        obj.is_fitted = True
        return obj
