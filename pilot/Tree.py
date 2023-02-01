""" This is an implementation of the tree data structure"""
class tree(object):
    """
    We use a tree object to save the PILOT model.

    Attributes:
    -----------

    node: str,
        type of the regression model
        'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
    pivot: tuple,
        a tuple to indicate where we performed a split. The first
        coordinate is the feature_id and the second one is
        the pivot.
    lm_l: ndarray,
        a 1D array to indicate the linear model for the left child node. The first element
        is the coef and the second element is the intercept.
    lm_r: ndarray,
        a 1D array to indicate the linear model for the right child node. The first element
        is the coef and the second element is the intercept.
    Rt: float,
        a real number indicating the rss in the present node.
    depth: int,
        the depth of the current node/subtree
    interval: ndarray,
        1D float array for the range of the selected predictor in the training data
    pivot_c: ndarry,
        1D int array. Indicating the levels in the left node
        if the selected predictor is categorical
    """

    def __init__(
        self,
        node=None,
        pivot=None,
        lm_l=None,
        lm_r=None,
        Rt=None,
        depth=None,
        interval=None,
        pivot_c=None,
    ) -> None:
        """
        Here we input the tree attributes.

        parameters:
        ----------
        node: str,
            type of the regression model
            'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
        pivot: tuple,
            a tuple to indicate where we performed a split. The first
            coordinate is the feature_id and the second one is
            the pivot.
        lm_l: ndarray,
            a 1D array to indicate the linear model for the left child node. The first element
            is the coef and the second element is the intercept.
        lm_r: ndarray,
            a 1D array to indicate the linear model for the right child node. The first element
            is the coef and the second element is the intercept.
        Rt: float,
            a real number indicating the rss in the present node.
        depth: int,
            the depth of the current node/subtree
        interval: ndarray,
            1D float array for the range of the selected predictor in the training data
        pivot_c: ndarry,
            1D int array. Indicating the levels in the left node
            if the selected predictor is categorical

        """
        self.left = None  # go left by default if node is 'lin'
        self.right = None
        self.Rt = Rt
        self.node = node
        self.pivot = pivot
        self.lm_l = lm_l
        self.lm_r = lm_r
        self.depth = depth
        self.interval = interval
        self.pivot_c = pivot_c

    def nodes_selected(self, depth=None):
        """
        count the number of models selected in the tree

        parameters:
        -----------
        depth: int, default = None.
            If specified count the number of models until the
            specified depth.
        """

        if self.node == "END":
            return {"con": 0, "lin": 0, "blin": 0, "pcon": 0, "plin": 0}
        elif depth is not None and self.depth == depth + 1:
            # find the first node that reaches depth + 1
            return {"con": 0, "lin": 0, "blin": 0, "pcon": 0, "plin": 0}
        nodes_l = self.left.nodes_selected(depth)
        nodes_l[self.node] += 1
        if self.node in ["plin", "pcon"]:
            nodes_r = self.right.nodes_selected(depth)
            for k in nodes_l.keys():
                nodes_l[k] += nodes_r[k]
        return nodes_l