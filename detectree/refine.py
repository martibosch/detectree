"""Methods to refine the pixel-level classification."""

import maxflow as mf
import numpy as np

MOORE_NEIGHBORHOOD_ARR = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])


def maxflow_refine(
    p_tree_img,
    tree_val,
    nontree_val,
    *,
    refine_int_rescale=10000,
    refine_beta=50,
):
    """Refine the pixel-level classification using a graph max-flow algorithm.

    Parameters
    ----------
    p_tree_img : numpy.ndarray
        The probability image of the pixel being a tree, as a two-dimensional numpy
        array with values between 0 and 1.
    tree_val, nontree_val : int, optional
        The values that designate tree and non-tree pixels respectively in the output
        array.
    refine_int_rescale : int, optional
        Parameter of the refinement procedure that controls the precision of the
        transformation of float to integer edge weights, required for the employed
        graph cuts algorithm. Larger values lead to greater precision.
    refine_beta : int, optional
        Parameter of the refinement procedure that controls the smoothness of the
        labelling. Larger values lead to smoother shapes.

    Returns
    -------
    img : numpy.ndarray
        The refined pixel-level classification as a two-dimensional numpy array with
        the same shape as `p_tree_img`.
    """
    # p_nontree, p_tree = np.hsplit(p_pred, 2)
    g = mf.Graph[int]()
    node_ids = g.add_grid_nodes(p_tree_img.shape)
    # P_nontree = p_nontree.reshape(img_shape)
    # P_tree = p_tree.reshape(img_shape)

    # The classifier probabilities are floats between 0 and 1, and the graph
    # cuts algorithm requires an integer representation. Therefore, we multiply
    # the probabilities by an arbitrary large number and then transform the
    # result to integers. For instance, we could use a `refine_int_rescale` of
    # `100` so that the probabilities are rescaled into integers between 0 and
    # 100 like percentages). The larger `refine_int_rescale`, the greater the
    # precision.
    # ACHTUNG: the data term when the pixel is a tree is `log(1 - P_tree)`,
    # i.e., `log(P_nontree)`, so the two lines below are correct
    D_tree = (refine_int_rescale * np.log(1 - p_tree_img)).astype(int)
    D_nontree = (refine_int_rescale * np.log(p_tree_img)).astype(int)
    # TODO: option to choose Moore/Von Neumann neighborhood?
    g.add_grid_edges(node_ids, refine_beta, structure=MOORE_NEIGHBORHOOD_ARR)
    g.add_grid_tedges(node_ids, D_tree, D_nontree)
    g.maxflow()
    # y_pred = g.get_grid_segments(node_ids)
    # transform boolean `g.get_grid_segments(node_ids)` to an array of
    # `self.tree_val` and `self.nontree_val`
    y_pred = np.full_like(p_tree_img, nontree_val)
    y_pred[g.get_grid_segments(node_ids)] = tree_val

    return y_pred
