try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logger.info("Google Generative AI SDK not found. Google features will be disabled.")