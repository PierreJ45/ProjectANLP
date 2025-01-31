def get_unicode(char):
    """
    Returns the Unicode code point of a given character.

    Args:
    char (str): A single character.

    Returns:
    str: The Unicode code point in the format 'U+XXXX'.
    """
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    return f"U+{ord(char):04X}"
