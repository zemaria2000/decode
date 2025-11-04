"""
    Sub-class for classification problems
"""

# xai_tree/regression.py
import numpy as np
from .base_tree import BaseTree
from .utils import log_loss, gini
from typing import Union
from .plotting import plot_classification_tree
import matplotlib.pyplot as plt
from .logger import logger
import os

class ClassificationTree(BaseTree):

    def __init__(self, model, scalerX = None, scalerY = None, dataset: np.ndarray = None, targets: np.ndarray = None, feature_names: list = None, target_names: list = None, feature_types: list = None, seed: int = 42):
        super().__init__(
            model = model,
            scalerX = scalerX,
            scalerY = scalerY,
            dataset = dataset,
            targets = targets,
            feature_names = feature_names,
            target_names = target_names,
            feature_types = feature_types,
            seed = seed
        )

    # Auxiliary methods to conduct bin refinement 
    def _refine_step_bins(self, feature_data: np.ndarray, preds: np.ndarray, bin_idx: np.ndarray, edges: np.ndarray):
        """
            Function that will conduct bin refinement (when we want an adaptive discretization process)
            Args:
                - feature_data (np.ndarray): values for the current feature being assessed
                - preds (np.ndarray): current predictions conducted by the model
                - bin_idx (np.ndarray): array with info, for each predicted sample, on their current bin for the current discretization
                - edges (np.ndarray): all current discretization values
            Return:
                - refine_bins (np.ndarray): array with the bin indexes to be refined
        """
        # NOTE: STILL IN DEVELOPMENT!
        bin_vars = [np.var(preds[bin_idx == i], ddof=1) if np.sum(bin_idx == i) > 1 else np.nan for i in range(len(edges)-1)]
        refine_bins = np.where(bin_vars > np.median(bin_vars) * 1.5)[0]

        return refine_bins
    
    # Calculating impurity
    def _impurity_calculation(self, preds: np.ndarray, criterium: str = 'default'):
        """
            Impurity calculation based on the criterium chosen
            Args:
                - preds (np.ndarray): predictions conducted by the model
                - criterium (str): default, or 'gini', 'log_loss'
            Returns:
                - score (float): the target candidate score
        """
        # Auxiliary step for calculating the prediction distribution
        preds_dist = np.bincount(preds, minlength=len(self.target_names))
        if criterium == 'default' or criterium == 'log_loss':
            return log_loss(preds_dist)
        elif criterium == 'gini':
            return gini(preds_dist)


    def _initialise_root_node(self, final_res):
        """
            Initialising the root node information
            Args:
                - final_res (tuple): tuple that comprises the best candidate score, in the form of (feature_idx, feature_value, candidate_score)
        """
        # Auxiliary variable - class distribution for the original targets
        target_dist = np.unique(self.targets, return_counts = True)[1]

        ## Appending the necessary information
        self.N_NODE_SAMPLES.append(len(self.dataset))
        self.IMPURITY.append(gini(target_dist))
        self.VALUE.append(final_res[1])
        self.FEATURE.append(int(final_res[0]))
        self.LEFT.append(0)                      
        self.DEPTH.append(0)           
        self.LEAF.append(0)            
        self.PARENT.append(-1)
        self.CLASS.append(np.argmax(target_dist))
        self.CLASS_DIST.append(target_dist)
        self.NODE_DS.append(self.dataset.copy())    


    def _expand_node(self, 
                     prev_node_idx: int, 
                     current_depth: int, 
                     dataset: np.ndarray, 
                     imp_threshold: float, 
                     n_steps: Union[str, int], 
                     adaptive: bool, 
                     criterium: str, 
                     perturb_dataset: bool, 
                     left: bool):
        """
            Creating a new node based on the previous
            Args:  
                - prev_node_idx (int): parent node index
                - current_depth (int): current depth in the tree
                - dataset (np.ndarray): current dataset being evaluated
                - imp_threshold (float): threshold that defines a stopping criteria
                - n_steps (Union[str, int]): steps approach for discretization
                - adaptive (bool): whether to implement adaptive discretization
                - criterium (str): criterium for loss assessment in the tree
                - perturb_dataset (bool): if we want to perturb dataset during step creation
                - left (bool): if we are in a left (True) or a right (False) node
        """
        logger.debug(f"Expanding node {prev_node_idx} (depth={self.DEPTH[prev_node_idx]}) → "
             f"feature={self.feature_names[self.FEATURE[prev_node_idx]]}, threshold={self.VALUE[prev_node_idx]:.3f}")

        # 0. Some initial base information regarding the node
        self.LEFT.append(1 if left else 0)
        self.PARENT.append(prev_node_idx)
        self.DEPTH.append(current_depth)
        self.NODE_DS.append(dataset)

        # 1. Conducting predictions with the model
        ds = self._scale_inputs(dataset)
        preds = self.model.predict(ds)
        preds_dist = np.bincount(preds, minlength = len(self.target_names))
        # Appending some more info about the node
        node_imp = gini(preds_dist)
        self.N_NODE_SAMPLES.append(len(preds))
        self.IMPURITY.append(node_imp)
        self.CLASS.append(np.argmax(preds_dist))
        self.CLASS_DIST.append(preds_dist)

        # 1st verification - impurity threshold or max depth
        if node_imp < imp_threshold or current_depth == self.max_depth:
            self.LEAF.append(1)
            self.VALUE.append(None)
            self.FEATURE.append(None)
            if current_depth == self.max_depth:
                logger.debug(f"Max depth reached at node {len(self.FEATURE)-1} -> leaf node")
            if node_imp < imp_threshold:
                logger.debug(f"Stopping split: impurity {node_imp:.4f} < threshold {imp_threshold}")
        else:
            # Conduct the optimisation process for the rule
            step_info = self._create_auto_steps(dataset, n_steps = n_steps, adaptive = adaptive)
            score_list = self.execute_trial(dataset, step_info, criterium = criterium, perturb_dataset = perturb_dataset)
            final_res = self.implement_choice(score_list)
            # In case no target rule was found
            if final_res is None:
                logger.warning(f"No acceptable candidate split was found for node {prev_node_idx}. Marking it as a leaf node")
                self.LEAF.append(1)
                self.VALUE.append(None)
                self.FEATURE.append(None)            
            else:
                logger.debug(f"Candidate split found for node {prev_node_idx}: Rule is {int(final_res[0])} <= {final_res[1]}")
                self.LEAF.append(0)
                self.VALUE.append(final_res[1])
                self.FEATURE.append(int(final_res[0]))
        logger.debug(f"Details: number of samples = {self.N_NODE_SAMPLES[-1]}, impurity = {self.IMPURITY[-1]}, output class = {self.CLASS[-1]}, class distribution = {self.CLASS_DIST[-1]}")
    
    def _return_leaf_output(self, node_idx: int):
        """
            Returns the output class for the current prediction (current node)
            Args:
                - node_idx (int): current index
        """
        return self.CLASS[node_idx]
    

    def plot_tree(self, save_fig: bool = False, file_name: str = None):
        """ 
            Plotting the classification tree
            Args:
                - save_fig (bool): whether the user wants, or not, to save the figure
                - file_name (str): name of the file to be stored (no need for the extension, I define pdf as default)
        """
        plot_classification_tree(self)
                
        if save_fig and file_name is not None:
            plt.savefig(os.path.join(self.storage.folders.figs, f'{file_name}.pdf'), dpi = 300)
        # plt.show()


    def extract_prediction_path(self, sample: np.array):
        """
            Function that will extract the prediction path for each sample in the test set
            Args:
                - sample (np.array): sample to be analysed
            Returns:
                - text (str): textual explanation of the prediction path for the sample

            Code inspiration:"
            "https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html""
        """

        # 1. Making a prediction for our sample
        pred, path = self.make_predictions(sample) 

        # Auxiliary step
        sample = np.array(sample).reshape(1, -1)

        # 2. Extracting relevant information from the tree to generate the text 
        nodes = np.array(path).squeeze()
        feature_indices = np.array(self.FEATURE).squeeze()[nodes]
        feature_values = np.round(np.array(self.VALUE).squeeze()[nodes], 3)
        left_right = np.array(self.LEFT).squeeze()[nodes[1:]] 
        left_right = ['left' if val == 1 else 'right' for val in left_right]
        final_pred_str = self.target_names[pred[0]]
        final_class_dist = np.array(self.CLASS_DIST).squeeze()[nodes[-1]]
        certainty = final_class_dist[pred[0]] / np.sum(final_class_dist)

        # 3. Generating a textual explanation for the prediction path

        if len(nodes) <= 2:
            # Initial rule == only rule in fact
            text = f"\n\nThe decision tree starts at the root node, with the initial and only rule for this node being: \033[3mif {self.feature_names[feature_indices[0]]} \u2264 {feature_values[0]}, move to the left child; otherwise, move to the right\033[0m.\n    The test sample has value {self.feature_names[feature_indices[0]]} = {sample[0, feature_indices[0]]}, so it reaches the leaf node {nodes[1]}.\n\n"
            # Final text for the prediction
            text += f"\033[1mThe final prediction is {final_pred_str}, with a certainty of {np.round(100*certainty, 2)}%.\033[0m\n"

        else:

            # Initial text for the initial rule
            text = f"\n\nThe decision tree starts at the root node, with the initial rule being: \033[3mif {self.feature_names[feature_indices[0]]} \u2264 {feature_values[0]}, move to the left child; otherwise, move to the right\033[0m.\n    The test sample has value {self.feature_names[feature_indices[0]]} = {sample[0, feature_indices[0]]}, so it moves to the {left_right[0]} child (node {nodes[1]}).\n"

            # Cycle for the middle rules
            for i, (index, feature_index, rule_feature_value) in enumerate(zip(nodes[1:-2], feature_indices[1:-2], feature_values[1:-2])):
                text += f"At node {index}, the rule is: \033[3mif {self.feature_names[feature_index]} \u2264 {rule_feature_value}, move to the left child; otherwise, move to the right\033[0m.\n    The test sample has value {self.feature_names[feature_index]} = {sample[0, feature_index]}, so it moves to the {left_right[i+1]} child (node {nodes[i+2]}).\n"

            # Final text for the last rule
            text += f"At node {nodes[-2]}, the last decision node for this prediction, the final rule is: \033[3mif {self.feature_names[feature_indices[-2]]} \u2264 {feature_values[-2]}, move to the left child; otherwise, move to the right\033[0m.\n     The test sample has value {self.feature_names[feature_indices[-2]]} = {sample[0, feature_indices[-2]]}, so it reaches the leaf node {nodes[-1]}.\n\n"

            # Final prediction and sentence
            final_pred_str = self.target_names[pred[0]]
            text += f"\033[1mThe final prediction is {final_pred_str}, with a certainty of {np.round(100*certainty, 2)}%.\033[0m\n"

        return text
    
