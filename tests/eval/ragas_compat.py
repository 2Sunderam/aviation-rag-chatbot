"""Import shim so ragas 0.4.3 (latest) loads with langchain-community 0.4.

ragas imports VertexAI classes from langchain_community modules that were removed in 0.4.
It only uses them in isinstance() checks, so empty placeholder classes are safe.
Import this module BEFORE importing ragas. Used only by the evals, never by the app.
"""

import importlib
import sys
import types

_STUBS = {
    "langchain_community.chat_models.vertexai": ["ChatVertexAI"],
    "langchain_community.llms.vertexai": ["VertexAI"],
}

for mod_name, classes in _STUBS.items():
    try:
        importlib.import_module(mod_name)
    except ModuleNotFoundError:
        mod = types.ModuleType(mod_name)
        for cls in classes:
            setattr(mod, cls, type(cls, (), {}))
        sys.modules[mod_name] = mod
