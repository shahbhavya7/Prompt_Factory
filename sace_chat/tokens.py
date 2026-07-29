try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None


def est_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return int(len(text.split()) / 0.75)
