try:
    from recommender.LLMHelper import LLMHelper
except Exception as e:
    # Fallback: define a dummy class so imports don't crash the app
    class LLMHelper:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"LLMHelper could not be imported: {e}")

__all__ = ["LLMHelper"]
