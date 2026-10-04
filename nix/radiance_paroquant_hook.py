"""Lazy registration shim for the radiance ParoQuant quant method.

Installed in vLLM's site-packages and imported by a `.pth` side effect at
interpreter start. Importing vLLM here, during site initialization, caches the
platform before the ROCm plugin is ready; instead this installs an import hook
that imports `radiance_paroquant` after `vllm.config` has fully initialized,
at which point the quantization registry exists and `get_quantization_config`
can resolve `quant_method: paroquant`. The import is best-effort so unrelated
models still load if the ParoQuant kernel is unavailable."""

import sys
import warnings


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        if "radiance_paroquant" in sys.modules:
            return
        try:
            import radiance_paroquant  # noqa: F401
        except Exception as error:  # noqa: BLE001 - optional feature
            warnings.warn("[radiance] paroquant registration failed: %r" % (error,))


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
