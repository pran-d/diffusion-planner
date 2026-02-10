class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)
