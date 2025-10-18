from typing import List, Dict, Any, Tuple
class BaseAddOn:
    """Abstract base class for modular FeaturePipeline Add-ons."""
    # ------------------- Data Processing Stages ------------------- #
    def before_sequence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs before data is converted to sequences.
        Typical use: cleaning, normalization, or fitting scalers.

        Args:
            state (Dict): Contains at least 'df_data' and 'df_labels'.
        Returns:
            Dict: Updated state.
        """
        return state

    def apply_window(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs after data is windowed into sequences.
        Typical use: per-window feature engineering or filtering.

        Args:
            state (Dict): Contains 'X_list', 'y_list', etc.
        Returns:
            Dict: Updated state.
        """
        return state

    def transformation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs after windows have been applied.
        Typical use: reshaping or concatenating sequence data.

        Args:
            state (Dict): Contains 'X_list', 'y_list', etc.
        Returns:
            Dict: Updated state.
        """
        return state

    # ------------------- Evaluation Stages ------------------- #
    def on_evaluation(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called during evaluation, before metric computation.
        Typical use: inverse scaling of predictions and labels.

        Args:
            eval_data (Dict): Contains:
                - 'all_preds_reg': np.ndarray of predictions
                - 'all_labels_reg': np.ndarray of labels
                - 'metrics': dict of current metrics (may be empty)
        Returns:
            Dict: Updated eval_data (preds/labels may be transformed)
        """
        return eval_data

    def on_evaluation_end(self, eval_data: Dict[str, Any]) -> None:
        """
        Called after metrics are computed.
        Typical use: final reporting, visualization, or logging.

        Args:
            eval_data (Dict): Same dict as in `on_evaluation`.
        """
        pass
