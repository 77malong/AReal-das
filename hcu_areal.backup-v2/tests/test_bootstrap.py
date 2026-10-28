import importlib
import pathlib
import sys
import tempfile
import unittest
from hcu_areal import bootstrap
from hcu_areal.patch_utils import AReaLPatchesManager

class TestBootstrap(unittest.TestCase):
    def test_lazy_import_patch_on_new_module(self):
        name="hcu_areal_test_target_module"
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, name+".py").write_text(
                "class Engine:\n    def run(self): return 3\n", encoding="utf-8")
            sys.path.insert(0,tmp)
            importlib.invalidate_caches()
            mgr=AReaLPatchesManager()
            mgr.register_patch(name+".Engine.run", lambda original: (lambda self: original(self)*2),
                               module_name=name, patch_id="test.lazy", apply_wrapper=True)
            try:
                bootstrap.install(mgr)
                module=importlib.import_module(name)
                self.assertEqual(module.Engine().run(),6)
                bootstrap.uninstall()
                self.assertEqual(module.Engine().run(),3)
            finally:
                bootstrap.uninstall()
                sys.modules.pop(name,None)
                sys.path.remove(tmp)

    def test_default_is_off(self):
        from hcu_areal.areal_adaptor import patch_features
        mgr=patch_features(env={})
        self.assertEqual(mgr.patch_ids, ())
        mgr=patch_features(env={"HCU_AREAL_PATCH_QWEN35_AWEX":"1"})
        self.assertEqual(set(mgr.patch_ids), {"qwen35.awex.trace.train", "qwen35.awex.trace.infer"})
