class BaseAugs:
    def __init__(self):
        self.augs = self.get_augs()
    def get_augs(self):
        return None
    def __call__(self, *args, **kwargs):
        return self.augs(*args, **kwargs)
