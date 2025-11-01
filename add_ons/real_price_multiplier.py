import numpy as np
from typing import Dict, Any
from add_ons.base_addon import BaseAddOn

class RealPriceMultiplier(BaseAddOn):
    """
    AddOn to scale normalized/multiplied predictions back to real prices
    using the last close price.

    This addon assumes your model's predictions (and corresponding labels) 
    are multipliers (e.g., 1.05 for +5%) relative to the last close price.
    """
    def __init__(self):
        self.on_evaluation_priority = 1
        
    def on_evaluation(
        self, 
        eval_data: Dict[str, Any], 
        pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Called by `evaluate_model` *before* final MSE/MAE calculation.
        
        Scales 'all_preds_reg' and 'all_labels_reg' using 'all_last_close_prices'.
        """
        print("➡️ Running RealPriceMultiplier on_evaluation...")
        
        # 1. Get data from eval_data
        preds = eval_data.get("all_preds_reg")
        labels = eval_data.get("all_labels_reg")
        last_prices = eval_data.get("all_last_close_prices")

        # 2. Validate
        if preds is None or labels is None or last_prices is None:
            print("  ⚠️ RealPriceMultiplier: Missing 'all_preds_reg', 'all_labels_reg', or 'all_last_close_prices' in eval_data. Skipping.")
            return eval_data
            
        if not (len(preds) == len(labels) == len(last_prices)):
            print(f"  ⚠️ RealPriceMultiplier: Mismatched lengths. Preds: {len(preds)}, Labels: {len(labels)}, Prices: {len(last_prices)}. Skipping.")
            return eval_data

        # 3. Scale the values in-place
        # This performs element-wise multiplication
        eval_data["all_preds_reg"] = preds * last_prices
        eval_data["all_labels_reg"] = labels * last_prices
        
        print("  ✅ RealPriceMultiplier: Scaled predictions and labels to real prices.")
        return eval_data

    def on_server_inference(
        self, 
        inference_payload: Dict[str, Any], 
        pipeline_extra_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Called by the server/inference endpoint.
        
        Scales 'y_pred_np' using 'last_close_price' provided in the payload.
        """
        # 1. Get data from payload
        preds = inference_payload.get("y_pred_np")
        last_price = inference_payload.get("last_close_price") # Must be added by the caller

        # 2. Validate
        if preds is None:
            print("  ⚠️ RealPriceMultiplier: Missing 'y_pred_np' in inference_payload. Skipping.")
            return inference_payload
            
        if last_price is None:
            print("  ⚠️ RealPriceMultiplier: Missing 'last_close_price' in inference_payload. Skipping.")
            print("     (Note: The inference endpoint must add 'last_close_price' to the payload *before* calling run_on_server_inference).")
            return inference_payload

        # 3. Scale the values in-place
        inference_payload["y_pred_np"] = preds * last_price
        
        print("  ✅ RealPriceMultiplier: Scaled inference predictions to real prices.")
        return inference_payload