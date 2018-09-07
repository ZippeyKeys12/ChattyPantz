# From: http://zachmoshe.com/2017/04/03/pickling-keras-models.html

import tempfile
import types

import tensorflow.keras.models as kmodels


def __getstate__(self):
    model_str = ""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
        kmodels.save_model(self, fd.name, overwrite=True)
        model_str = fd.read()
    d = {"model_str": model_str}
    return d


def __setstate__(self, state):
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
        fd.write(state["model_str"])
        fd.flush()
        model = kmodels.load_model(fd.name)
    self.__dict__ = model.__dict__


cls = kmodels.Model
cls.__getstate__ = __getstate__
cls.__setstate__ = __setstate__
