def traceable(stage_name: str, loop: bool = False):
    """
    Marks a FeaturePipeline method as traceable.
    stage_name: Display name for table
    loop: Whether this stage calls multiple add-ons internally
    """
    def decorator(func):
        func._trace_meta = {"stage_name": stage_name, "loop": loop}
        return func
    return decorator