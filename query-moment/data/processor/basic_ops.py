from kn_util.general import global_registry


@global_registry.register_processor("delete")
class Delete:
    def __init__(self, from_keys=None) -> None:
        assert from_keys
        self.from_keys = from_keys

    def __call__(self, result):

        for k in self.from_keys:
            del result[k]
        return result


@global_registry.register_processor("rename")
class Rename:
    def __init__(self, from_keys=None, to_keys=None) -> None:
        assert from_keys and to_keys
        self.from_keys = from_keys
        self.to_keys = to_keys

    def __call__(self, result):
        _result = dict()
        for from_key, to_key in zip(self.from_keys, self.to_keys):
            _result[to_key] = result[from_key]
            del result[from_key]
        result.update(_result)

        return result

@global_registry.register_processor("lambda")
class Lambda:
    def __init__(self, _lambda, to_key):
        self._lambda=_lambda
        self.to_key = to_key
    
    def __call__(self, result):
        result[self.to_key] = self._lambda(result)
        return result


@global_registry.register_processor("collect")
class Collect:
    def __init__(self, from_keys=[]):
        self.from_keys = from_keys

    def __call__(self, result):
        _result = dict()
        for k in self.from_keys:
            _result[k] = result[k]
        return _result
