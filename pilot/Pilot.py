""" This module implements PILOT"""
import numba as nb
from numba import jit
import numpy as np
import pandas as pd
from .Tree import tree

@nb.njit(parallel=True)
def isin(a, b):
    """
    For each element in 'a' compute if it is in 'b'

    parameters:
    ----------
    a: ndarray, 1D float array. An integer array of indices.
    b: ndarray, 1D float array. An integer array of indices.

    returns:
    ------
    ndarray: 1D boolean array, indicating if the elements in 'a' is in 'b'.
    """
    out = np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out


@jit(nb.float64[:](nb.types.unicode_type, nb.int64, nb.float64[:], nb.int64[:]))
def loss_fun(criteria, num, Rss, k):
    """
    This function is used to compute the information criteria

    parameters:
    ----------
    criteria: str,
        the information criteria
    num: int,
        total number of samples
    Rss: float,
        the residual sum of squares, can be a vector
    k: ndarray,
        1D int array to describe the degrees of freedom, can be a vector

    return:
    -------
    float: The loss according to the information criteria
    """

    if criteria == "AIC":
        return num * np.log(Rss / num) + 2 * k
    elif criteria == "AICc":
        return num * np.log(Rss / num) + 2 * k + (2 * k**2 + 2 * k) / (num - k - 1)
    elif criteria == "BIC":
        return num * np.log(Rss / num) + np.log(num) * k
    return np.array([0.0])


@jit(
    nb.types.Tuple(
        (
            nb.int64,
            nb.float64,
            nb.types.unicode_type,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.int64[:],
        )
    )(
        nb.int64[:],
        nb.typeof(["a", "b"]),
        nb.int64,
        nb.int64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.types.unicode_type,
        nb.int64,
        nb.int64[:],
        nb.int64[:],
    ),
    nopython=True,
)
def best_split(
    index,
    regression_nodes,
    n_features,
    sorted_X_indices,
    X,
    y,
    split_criterion,
    min_sample_leaf,
    k,
    categorical,
):
    """
    This function finds the best split as well as the linear
    model on the node.

    parameters:
    -----------
    index: ndarray,
        1D int array. the indices of samples in the current node, which
        would induce sorted_X and sorted_y.
    regression_nodes: list,
        a list indicating which kinds of node is used,
        we have 'lin'/'pcon'/'blin'/'plin'.
    n_features: int,
        number of features.
    sorted_X_indices: ndarray,
        2D int array. Sorted indices of cases, according to each feature.
    X: ndarray,
        2D float array, the predictors.
    y: ndarray,
        2D float array, the response.
    split_criterion: str,
        the criterion to split the tree,
        default is 'BIC', can also be 'AIC'/'AICc', etc.
    min_sample_leaf: int,
        the minimal number of samples required
        to be at a leaf node
    k: ndarray,
        1D int array, the degrees of freedom for 'pcon'/'blin'/'plin'.
    categorical: ndarray,
        1D int array, the columns of categorical variable, array.

    returns:
    --------
    best_feature: int,
        The feature/predictor id at which the dataset is best split.
        if it is a categorical feature, the second element is a list of values
        indicating the left region.
    best_pivot: float,
        The feature id at which the dataset is best split.
    best_node: str,
        The best regression model.
    lm_L: ndarray,
        1D float array. The linear model on the left node (intercept, coeficents).
    lm_R:  ndarray,
        1D float array. The linear model on the right node (intercept, coeficents).
        for 'lin' and 'con': lm_R is None, all information is included in lm_L
    interval: ndarray,
        1D float array. The range of the training data on this node
    pivot_c: ndarray,
        1D int array. An array of the levels belong to the left node.
        Used if the chosen feature/predictor is categorical.

    Remark:
    -------
    If the input data is not allowed to split, the function will return default
    values.
    """

    # Initialize variables, should be consistent with the variable type
    best_pivot = -1.0
    best_node = ""
    best_loss = -1.0
    best_feature = -1
    lm_L = np.array([0.0, 0.0])
    lm_R = np.array([0.0, 0.0])
    interval = np.array([-np.inf, np.inf])
    pivot_c = np.array([0])

    # Initialize the coef and intercept for 'blin'/'plin'/'pcon'
    l = (
        1 * ("blin" in regression_nodes)
        + 1 * ("plin" in regression_nodes)
        + 1 * ("pcon" in regression_nodes)
    )
    coef = np.zeros((l, 2)) * np.nan
    intercept = np.zeros((l, 2)) * np.nan

    # search for the best split among all features, negelecting the indices column
    for feature_id in range(1, n_features + 1):
        # get sorted X, y
        idx = sorted_X_indices[feature_id - 1]
        idx = idx[isin(idx, index)]
        X_sorted, y_sorted = X[idx].copy(), y[idx].copy()

        # Initialize possible pivots
        possible_p = np.unique(X_sorted[:, feature_id])
        lenp = len(possible_p)

        if feature_id - 1 not in categorical:
            num = np.array([0, X_sorted.shape[0]])

            # store entries of the Gram and moment matrices
            Moments = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        np.sum(X_sorted[:, feature_id]),
                        np.sum(X_sorted[:, feature_id] ** 2),
                        np.sum(
                            X_sorted[:, feature_id].copy().reshape(-1, 1) * y_sorted
                        ),
                        np.sum(y_sorted),
                        np.sum(y_sorted**2),
                    ],
                ]
            )

            # CON:
            if "con" in regression_nodes:
                intercept_con = Moments[1, 3] / num[1]
                coef_con = 0
                # compute the RSS and the loss according to the information criterion
                rss = (
                    Moments[1, 4]
                    + (num[1] * intercept_con**2)
                    - 2 * intercept_con * Moments[1, 3]
                )
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss]),
                    k=np.array([1], dtype=np.int64),
                )
                # update best_loss immediately
                if best_node == "" or loss.item() < best_loss:
                    best_node = "con"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_con, intercept_con])

            # LIN:
            if "lin" in regression_nodes and lenp >= 5:
                var = num[1] * Moments[1, 1] - Moments[1, 0] ** 2
                # in case a constant feature
                if var == 0:
                    coef_lin = 0
                else:
                    coef_lin = (
                        num[1] * Moments[1, 2] - Moments[1, 0] * Moments[1, 3]
                    ) / var
                intercept_lin = (Moments[1, 3] - coef_lin * Moments[1, 0]) / num[1]
                # compute the RSS and the loss according to the information criterion
                rss = (
                    Moments[1, 4]
                    + (num[1] * intercept_lin**2)
                    + (2 * coef_lin * intercept_lin * Moments[1, 0])
                    + coef_lin**2 * Moments[1, 1]
                    - 2 * intercept_lin * Moments[1, 3]
                    - 2 * coef_lin * Moments[1, 2]
                )
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss]),
                    k=np.array([2], dtype=np.int64),
                )
                # update best_loss immediately
                if best_loss == "" or loss.item() < best_loss:
                    best_node = "lin"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_lin, intercept_lin])

            # For blin, we need to maintain another Gram/moment matrices and the knot xi
            if "blin" in regression_nodes:
                # Moments need to be updated for blin:
                # [sum(x-xi)+, sum[(x-xi)+]**2, sum[x(x-xi)+], sum[y(x-xi)+]]
                XtX = np.array(
                    [
                        [
                            np.float64(num.sum()),
                            Moments[:, 0].sum(),
                            Moments[:, 0].sum(),
                        ],
                        [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                        [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                    ]
                )
                XtY = np.array([[Moments[1, 3]], [Moments[1, 2]], [Moments[1, 2]]])
                pre_pivot = 0.0

            # pcon, blin and plin: try each possible split and
            # find the best one the last number are never used for split
            for p in range(possible_p.shape[0] - 1):
                # The pointer to select the column of coef and intercept
                i = 0
                pivot = possible_p[p]
                # update cases in the left region
                index_add = X_sorted[:, feature_id] == pivot
                X_add = X_sorted[index_add, feature_id]
                y_add = y_sorted[index_add]

                # BLIN:
                if "blin" in regression_nodes:
                    # First maintain xi
                    xi = pivot - pre_pivot

                    # update XtX and XtY
                    XtX += np.array(
                        [
                            [0.0, 0.0, -xi * num[1]],
                            [0.0, 0.0, -xi * Moments[1, 0]],
                            [
                                -xi * num[1],
                                -xi * Moments[1, 0],
                                xi**2 * num[1] - 2 * xi * XtX[0, 2],
                            ],
                        ]
                    )
                    XtY += np.array([[0.0], [0.0], [-xi * Moments[1, 3]]])

                    # useless to check the first pivot or partition that
                    # leads to less than min_sample_leaf samples
                    if (
                        pivot != possible_p[0]
                        and p >= 1
                        and lenp >= 5
                        and np.linalg.det(XtX) > 0.001
                        and num[0] + X_add.shape[0] >= min_sample_leaf
                        and num[1] - X_add.shape[0] >= min_sample_leaf
                    ):
                        coefs = np.linalg.solve(XtX, XtY).flatten()
                        coef[i, :] = np.array([coefs[1], coefs[1] + coefs[2]])
                        intercept[i, :] = np.array(
                            [coefs[0], coefs[0] - coefs[2] * pivot]
                        )
                    i += 1
                    pre_pivot = pivot

                # update num after blin is fitted
                num += np.array([1, -1]) * X_add.shape[0]

                # first update moments then check if this pivot is eligable for a pcon/plin split
                Moments_add = np.array(
                    [
                        np.sum(X_add),
                        np.sum(X_add**2),
                        np.sum(X_add.reshape(-1, 1) * y_add),
                        np.sum(y_add),
                        np.sum(y_add**2),
                    ]
                )
                Moments += Moments_add * np.array([[1.0], [-1.0]])

                # negelect ineligable split
                if num[0] < min_sample_leaf:
                    continue
                elif num[1] < min_sample_leaf:
                    break

                # 'pcon' fit
                if "pcon" in regression_nodes:
                    coef[i, :] = np.array([0, 0])
                    intercept[i, :] = (Moments[:, 3]) / num
                    i += 1

                # 'plin' for the first split candidate is equivalent to 'pcon'
                if (
                    "plin" in regression_nodes
                    and p >= 4
                    and lenp - p >= 5
                    and 0 not in num * Moments[:, 1] - Moments[:, 0] ** 2
                ):
                    # coef and intercept are vectors of dimension 1
                    # have to reshape X column in order to get correct cross product
                    # the intercept should be divided by the total number of samples
                    coef[i, :] = (
                        num * Moments[:, 2] - Moments[:, 0] * Moments[:, 3]
                    ) / (num * Moments[:, 1] - Moments[:, 0] ** 2)
                    intercept[i, :] = (Moments[:, 3] - coef[i, :] * Moments[:, 0]) / num

                # compute the rss and loss of the above 3 methods
                rss = (
                    Moments[:, 4]
                    + (num * intercept**2)
                    + (2 * coef * intercept * Moments[:, 0])
                    + coef**2 * Moments[:, 1]
                    - 2 * intercept * Moments[:, 3]
                    - 2 * coef * Moments[:, 2]
                ).sum(axis=1)

                # if no fit is done, continue
                if np.isnan(rss).all():
                    continue

                # update the best loss
                rss = np.maximum(10**-8, rss)
                loss = loss_fun(criteria=split_criterion, num=num.sum(), Rss=rss, k=k)

                if ~np.isnan(loss).all() and (
                    best_node == "" or np.nanmin(loss) < best_loss
                ):
                    best_loss = np.nanmin(loss)
                    index_min = np.where(loss == best_loss)[0].item()
                    add_index = 1 * ("lin" in regression_nodes) + 1 * (
                        "con" in regression_nodes
                    )
                    best_node = regression_nodes[add_index + index_min]
                    best_feature = feature_id  # asigned but will not be used for 'lin'
                    interval = np.array([possible_p[0], possible_p[-1]])
                    best_pivot = pivot
                    lm_L = np.array([coef[index_min, 0], intercept[index_min, 0]])
                    lm_R = np.array([coef[index_min, 1], intercept[index_min, 1]])

            continue

        # CATEGORICAL VARIABLES
        mean_vec = np.zeros(lenp)
        num_vec = np.zeros(lenp)
        for i in range(lenp):
            # mean values of the response w.r.t. each level
            mean_vec[i] = np.mean(y_sorted[X_sorted[:, feature_id] == possible_p[i]])
            # number of elements at each level
            num_vec[i] = y_sorted[X_sorted[:, feature_id] == possible_p[i]].shape[0]

        # sort unique values w.r.t. the mean of the responses
        mean_idx = np.argsort(mean_vec)
        num_vec = num_vec[mean_idx]
        sum_vec = mean_vec[mean_idx] * num_vec
        possible_p = possible_p[mean_idx]

        # loop over the sorted possible_p and find the best partition
        num = np.array([0.0, X_sorted.shape[0]])
        sum_all = np.array([0, np.sum(y_sorted)])
        for i in range(lenp - 1):
            # update the sum and num
            sum_all += np.array([1.0, -1.0]) * sum_vec[i]
            num += np.array([1.0, -1.0]) * num_vec[i]
            # find the indices of the elements in the left node
            sub_index = isin(X_sorted[:, feature_id], possible_p[: i + 1])
            # compute the rss
            rss = np.sum((y_sorted[sub_index] - sum_all[0] / num[0]) ** 2) + np.sum(
                (y_sorted[~sub_index] - sum_all[1] / num[1]) ** 2
            )
            rss = np.maximum(10**-8, rss)
            loss = loss_fun(
                criteria=split_criterion,
                num=num.sum(),
                Rss=np.array([rss]),
                k=np.array([5], dtype=np.int64),
            )
            if best_node == "" or loss.item() < best_loss:
                best_feature = feature_id
                best_node = "pconc"
                best_loss = loss.item()
                lm_L = np.array([0, sum_all[0] / num[0]])
                lm_R = np.array([0, sum_all[1] / num[1]])
                pivot_c = possible_p[: i + 1].copy()
                pivot_c = pivot_c.astype(np.int64)

    return best_feature, best_pivot, best_node, lm_L, lm_R, interval, pivot_c


class PILOT(object):
    """
    This is an implementation of the PILOT method.

    Attributes:
    -----------
    max_depth: int,
        the max depth allowed to grow in a tree.
    split_criterion: str,
        the criterion to split the tree,
        we have 'AIC'/'AICc'/'BIC'/'adjusted R^2', etc.
    regression_nodes: list,
        A list of regression models used.
        They are 'con', 'lin', 'blin', 'pcon', 'plin'.
    min_sample_split: int,
        the minimal number of samples required
        to split an internal node.
    min_sample_leaf: int,
        the minimal number of samples required
        to be at a leaf node.
    step_size: int,
        boosting step size.
    X: ndarray,
        2D float array of the predictors.
    y, y0: ndarray,
        2D float array of the responses.
    sorted_X_indices: ndarray,
        2D int array of sorted indices according to each feature.
    n_feature: int,
        number of features
    categorical: ndarray,
        1D int array indicating categorical predictors.
    model_tree: tree object,
        learned PILOT model tree.
    B1, B2: int
        upper and lower bound for the first truncation,
        learned from y.
    """

    def __init__(
        self,
        max_depth=12,
        split_criterion="BIC",
        min_sample_split=10,
        min_sample_leaf=5,
        step_size=1,
    ) -> None:
        """
        Here we input model parameters to build a tree,
        not all the parameters for split finding.

        parameters:
        -----------
        max_depth: int,
            the max depth allowed to grow in a tree.
        split_criterion: str,
            the criterion to split the tree,
            we have 'AIC'/'AICc'/'BIC'/'adjusted R^2', etc.
        min_sample_split: int,
            the minimal number of samples required
            to split an internal node.
        min_sample_leaf: int,
            the minimal number of samples required
            to be at a leaf node.
        step_size: int,
            boosting step size.
        """

        # initialize class attributes
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.regression_nodes = ["con", "lin", "blin", "pcon", "plin"]
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.step_size = step_size

        # attributes used for fitting
        self.X = None
        self.y = None
        self.y0 = None
        self.sorted_X_indices = None
        self.ymean = None
        self.n_features = None
        self.categorical = np.array([-1])
        self.model_tree = None
        self.B1 = None
        self.B2 = None

        rule = {"con": 0, "lin": 1, "blin": 2, "pcon": 3, "plin": 4}
        self.regression_nodes.sort(key=lambda x: rule[x])
        df = [5, 5, 7]
        k = {"blin": df[0], "pcon": df[1], "plin": df[2]}
        # used to store the degrees of freedom
        self.k = np.array(
            [k[key] for key in self.regression_nodes if key not in ["con", "lin"]],
            dtype=np.int64,
        )

    def stop_criterion(self, tree_depth, y):
        """
        Stop splitting when either the tree has reached max_depth or the number of the
        data in the leaf node is less than min_sample_leaf or the variance of the node
        is less than the threshold.

        parameters:
        -----------
        tree_depth: int,
            Current depth.
        y: ndarray,
            2D float array. The response variable.

        returns:
        --------
        boolean:
            whether to stop the recursion.

        """
        if tree_depth >= self.max_depth or y.shape[0] <= self.min_sample_split:
            return False
        return True

    def build_tree(self, tree_depth, indices, rss):
        """
        This function is based on the recursive algorithm. We keep
        growing the tree, until it meets the stopping criterion.
        The parameters root is to save the tree structure.

        parameters:
        -----------
        tree_depth: int,
            the depth of the tree. By definition, the depth
            of the root node is 0.
        indices: ndarray,
            1D array containing data with int type. It gives
            the indices of cases in this node, which will
            induce sorted_X.
        rss: float,
            The rss of the current node before fitting a model.

        return:
        -------
        tree object:
            If it meets stop_criterion or can not be further split,
            return end node (denoted by 'END').

        """

        tree_depth += 1
        # fit models on the node
        best_feature, best_pivot, best_node, lm_l, lm_r, interval, pivot_c = best_split(
            indices,
            self.regression_nodes,
            self.n_features,
            self.sorted_X_indices,
            self.X,
            self.y,
            self.split_criterion,
            self.min_sample_leaf,
            self.k,
            self.categorical,
        )  # find the best split
        # stop fitting the tree
        if best_node == "":
            return tree(node="END", Rt=rss)
        elif best_node in ["lin", "con"]:
            # do not include 'lin' and 'con' in the depth calculation
            tree_depth -= 1

        # build tree only if it doesn't meet the stop_criterion
        if self.stop_criterion(tree_depth, self.y[indices]):

            # define a new node
            # best_feature should - 1 because the 1st column is the indices
            node = tree(
                best_node,
                (best_feature - 1, best_pivot),
                lm_l,
                lm_r,
                Rt=rss,
                depth=tree_depth + 1,
                interval=interval,
                pivot_c=pivot_c,
            )

            # update X and y by vectorization, reshape them to make sure their sizes are correct
            if best_node == "lin":
                # unpdate y
                raw_res = self.y[indices] - self.step_size * (
                    lm_l[0] * self.X[indices, best_feature].reshape(-1, 1) + lm_l[1]
                )
                # truncate the prediction
                self.y[indices] = self.y0[indices] - np.maximum(
                    np.minimum(self.y0[indices] - raw_res, self.B1), self.B2
                )

                # recursion
                node.left = self.build_tree(
                    tree_depth,
                    indices,
                    np.maximum(
                        0, np.sum((self.y[indices] - np.mean(self.y[indices])) ** 2)
                    ),
                )

            elif best_node == "con":
                self.y[indices] -= self.step_size * (lm_l[1])

                # stop the recursion
                node.left = tree(node="END", Rt=np.sum(self.y[indices] ** 2))
                return node

            else:
                # find the indices for the cases in the left and right node
                if best_node == "pconc":
                    cond = isin(self.X[indices, best_feature], pivot_c)
                else:
                    cond = self.X[indices, best_feature] <= best_pivot
                indices_left = (self.X[indices][cond, 0]).astype(int)
                indices_right = (self.X[indices][~cond, 0]).astype(int)

                # compute the raw and truncated predicrtion
                rawres_left = (
                    self.y[indices_left]
                    - (
                        lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1)
                        + lm_l[1]
                    )
                ).copy()
                self.y[indices_left] = self.y0[indices_left] - np.maximum(
                    np.minimum(self.y0[indices_left] - rawres_left, self.B1), self.B2
                )
                rawres_right = (
                    self.y[indices_right]
                    - (
                        lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1)
                        + lm_r[1]
                    )
                ).copy()
                self.y[indices_right] = self.y0[indices_right] - np.maximum(
                    np.minimum(self.y0[indices_right] - rawres_right, self.B1), self.B2
                )

                # recursion
                node.left = self.build_tree(
                    tree_depth,
                    indices_left,
                    np.maximum(
                        0,
                        np.sum(
                            (self.y[indices_left] - np.mean(self.y[indices_left])) ** 2
                        ),
                    ),
                )
                node.right = self.build_tree(
                    tree_depth,
                    indices_right,
                    np.maximum(
                        0,
                        np.sum(
                            (self.y[indices_right] - np.mean(self.y[indices_right]))
                            ** 2
                        ),
                    ),
                )

        else:
            # stop recursion if meeting the stopping criterion
            return tree(node="END", Rt=rss)

        return node

    def fit(self, X, y, categorical=np.array([-1])):
        """
        This function is used for model fitting. It should return
        a pruned tree, which includes the location of each node
        and the linear model for it. The results should be saved
        in the form of class attributes.

        parameters:
        -----------
        X: Array-like objects, usually pandas.DataFrame or numpy arrays.
            The predictors.
        y: Array-like objects, usually pandas.DataFrame or numpy arrays.
            The responses.
        categorical: An array of column indices of categorical variables.
                     We assume that they are integer valued.

        return:
        -------
        None
        """

        # X and y should have the same size
        assert X.shape[0] == y.shape[0]

        # Switch pandas objects to numpy objects
        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)

        if isinstance(y, pd.core.frame.DataFrame):
            y = np.array(y)
        elif y.ndim == 1:
            y = y.reshape((-1, 1))

        # define class attributes
        self.n_features = X.shape[1]
        n_samples = X.shape[0]
        self.categorical = categorical

        # insert indices to the first column of X to memorize the indices
        self.X = np.c_[np.arange(0, n_samples, dtype=int), X]

        # Memorize the indices of the cases sorted along each feature
        # Do not sort the first column since they are just indices
        sorted_indices = np.array(
            [
                np.argsort(self.X[:, feature_id], axis=0).flatten()
                for feature_id in range(1, self.n_features + 1)
            ]
        )
        self.sorted_X_indices = (self.X[:, 0][sorted_indices]).astype(int)
        # ->(n_samples, n_features) 2D array

        # y should be remembered and modified during 'boosting'
        self.y = y.copy()  # calculate on y directly to save memory
        self.y0 = y.copy()  # for the truncation procudure
        self.B1 = (
            2 * y.max() - y.min()
        )  # compute the upper bound for the first truncation
        self.B2 = (
            -y.max() + 2 * y.min()
        )  # compute the lower bound for the second truncation

        # build the tree, only need to take in the indices for X
        self.model_tree = self.build_tree(
            -1, self.sorted_X_indices[0], np.sum((y - y.mean()) ** 2)
        )

        # if the first node is 'con'
        if self.model_tree.node == "END":
            self.ymean = y.mean()

        return

    def predict(self, model=None, x=None, maxd=np.inf):
        """
        This function is used for model predicting. Given a dataset,
        it will find its location and respective linear model.

        parameters:
        -----------
        model: The tree objects
        x: Array-like objects, new sample need to be predicted
        maxd: The maximum depth to be considered for prediction,
              can be less than the true depth of the tree.

        return:
        -------
        y_hat: numpy.array
               the predicted y values
        """
        y_hat = []
        if model is None:
            model = self.model_tree

        if isinstance(x, pd.core.frame.DataFrame):
            x = np.array(x)

        if self.model_tree.node == "END":
            return np.ones(x.shape[0]) * self.ymean

        for row in range(x.shape[0]):
            t = model
            y_hat_one = 0
            while t.node != "END" and t.depth < maxd:

                if t.node == "pconc":
                    if np.isin(x[row, t.pivot[0]], t.pivot_c):
                        y_hat_one += self.step_size * (t.lm_l[1])
                        t = t.left
                    else:
                        y_hat_one += self.step_size * (t.lm_r[1])
                        t = t.right

                # go left if 'lin'
                elif t.node in ["lin", "con"] or x[row, t.pivot[0]] <= t.pivot[1]:
                    if t.node == "lin":
                        # truncate both on the left and the right
                        y_hat_one += self.step_size * (
                            t.lm_l[0]
                            * np.min(
                                [
                                    np.max([x[row, t.pivot[0]], t.interval[0]]),
                                    t.interval[1],
                                ]
                            )
                            + t.lm_l[1]
                        )
                    else:
                        # truncate on the left
                        y_hat_one += self.step_size * (
                            t.lm_l[0] * np.max([x[row, t.pivot[0]], t.interval[0]])
                            + t.lm_l[1]
                        )
                    t = t.left

                else:
                    y_hat_one += self.step_size * (
                        t.lm_r[0] * np.min([x[row, t.pivot[0]], t.interval[1]])
                        + t.lm_r[1]
                    )
                    t = t.right

                # truncation
                if y_hat_one > self.B1:
                    y_hat_one = self.B1
                elif y_hat_one < self.B2:
                    y_hat_one = self.B2

            y_hat.append(y_hat_one)
        return np.array(y_hat)

    def print_tree(self, model_tree, level):
        """
        A function for tree visualization

        parameters:
        -----------
        """
        if model_tree is not None:
            self.print_tree(model_tree.left, level + 1)
            if model_tree.node == "lin":
                print(
                    " " * 8 * level + "-->",
                    model_tree.node,
                    (round(model_tree.pivot[0], 3)),
                    round(model_tree.Rt, 3),
                    (round(model_tree.lm_l[0], 3), round(model_tree.lm_l[1], 3)),
                    None,
                )
            elif model_tree.node == "END":
                print(" " * 8 * level + "-->" + "END", round(model_tree.Rt, 3))
            else:
                print(
                    " " * 8 * level + "-->",
                    model_tree.node,
                    (round(model_tree.pivot[0], 3), round(model_tree.pivot[1], 3)),
                    round(model_tree.Rt, 3),
                    (round(model_tree.lm_l[0], 3), round(model_tree.lm_l[1], 3)),
                    (round(model_tree.lm_r[0], 3), round(model_tree.lm_r[1], 3)),
                )
            self.print_tree(model_tree.right, level + 1)