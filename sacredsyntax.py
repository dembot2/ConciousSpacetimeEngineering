def seek_meaning(reality):
    if reality is None:
        return THE_TRUTH
    return seek_meaning(deconstruct(reality))