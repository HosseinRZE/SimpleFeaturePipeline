from typing import List, Dict, Any, Tuple
import tempfile, os, uuid, base64
from pathlib import Path
import pandas as pd
class BaseAddOn:

# ------------------- Data Processing Stages ------------------- #
    def before_sequence(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return state

    def apply_window(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return state

    def transformation(
        self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return state

    # ------------------- Evaluation Stages ------------------- #
    def on_evaluation(
        self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return eval_data

    def on_evaluation_end(
        self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return eval_data