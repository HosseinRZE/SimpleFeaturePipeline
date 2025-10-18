import joblib
import dill
import os

def save_pipeline(pipeline, pipeline_out):
    """
    Save the FeaturePipeline object to a file, using joblib or dill as fallback.

    Args:
        pipeline (FeaturePipeline): The pipeline object to be saved.
        pipeline_out (str): The output file path (including extension, e.g., .pkl or .dill).
    """
    try:
        # Attempt to save pipeline using joblib (preferred)
        joblib.dump(pipeline, pipeline_out)
        print(f"✅ Full FeaturePipeline saved to {pipeline_out}")
    except Exception as e:
        # Fallback to dill if joblib can't serialize some parts of the pipeline
        pipeline_out = pipeline_out.replace(".pkl", ".dill")  # Change extension to .dill
        with open(pipeline_out, "wb") as f:
            dill.dump(pipeline, f)