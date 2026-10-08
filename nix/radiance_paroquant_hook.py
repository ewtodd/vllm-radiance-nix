"""Lazy registration shim for the radiance ParoQuant quant methods.

Installed in vLLM's site-packages and imported by a `.pth` side effect at
interpreter start. Importing vLLM here, during site initialization, caches the
platform before the ROCm plugin is ready; instead this installs an import hook
that imports the ParoQuant modules after `vllm.config` has fully initialized,
at which point the quantization registry exists and `get_quantization_config`
can resolve `quant_method: paroquant`, `paroquant_mxfp4` and
`paroquant_mxfp6`. The imports are best-effort so unrelated models still load
if the ParoQuant kernels are unavailable."""

import importlib
import sys
import warnings


def _import_best_effort(module_name, label):
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - optional feature
        warnings.warn("[radiance] %s registration failed: %r" % (label, error))


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _import_best_effort("radiance_paroquant", "paroquant")
        _import_best_effort("radiance_paroquant_mxfp4", "paroquant_mxfp4/mxfp6")


class _Finder:
    target = "vllm.config"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        import importlib.machinery

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if not any(getattr(finder, "target", None) == _Finder.target for finder in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
