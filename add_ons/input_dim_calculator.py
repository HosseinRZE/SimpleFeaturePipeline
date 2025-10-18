from add_ons.base_addon import BaseAddOn 

class InputDimCalculator(BaseAddOn):
    def transformation(self, state: dict) -> dict:
        """
        Inspects the first sample of the final feature list (X_list)
        and calculates the input dimension for each feature group.
        """
        X_list = state.get('X_list')

        # If there's no data left after filtering, return an empty dict.
        if not X_list:
            state['input_dim'] = {}
            return state

        # Get the first sample to determine the structure.
        sample = X_list[0]

        if isinstance(sample, dict):
            # Case 1: Multi-input, features are in a dictionary.
            # e.g., {'main': tensor1, 'aux': tensor2}
            input_dim = {k: v.shape[1] for k, v in sample.items()}
        else:
            # Case 2: Single-input, features are a single tensor/array.
            input_dim = {"main": sample.shape[1]}
        
        # Add the calculated input_dim to the state for later use.
        state['input_dim'] = input_dim
        
        print(f"✅ Calculated input_dim: {input_dim}")
        return state