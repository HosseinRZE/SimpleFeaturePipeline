import os
import joblib
from utils.save_func_args import save_function_args
from utils.save_pipeline import save_pipeline

def save_model_files(
    base_dir,
    folder_name,
    trainer,
    model_out,
    meta_out,
    pipeline_out,
    pipeline,
    train_model,
    **train_args
):
    """
    Save model checkpoint, meta args, and pipeline into a unique folder.
    """

    # Create the experiment folder
    out_dir = os.path.join(base_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # Construct full paths for saving
    model_path = os.path.join(out_dir, model_out)
    meta_path = os.path.join(out_dir, meta_out)
    pipeline_path = os.path.join(out_dir, pipeline_out)

    # 1. Save model
    trainer.save_checkpoint(model_path)
    print(f"✅ Model saved to {model_path}")

    # 2. Save meta (function args)
    meta_args = save_function_args(train_model, **train_args)
    joblib.dump(meta_args, meta_path)
    print(f"✅ Meta saved to {meta_path}")

    # 3. Save pipeline
    save_pipeline(pipeline, pipeline_path)
