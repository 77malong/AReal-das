import importlib
import pathlib
import sys
import tempfile
import unittest
from hcu_areal import bootstrap
from hcu_areal.patch_utils import AReaLPatchesManager


class TestBootstrap(unittest.TestCase):
    def test_lazy_import_patch_on_new_module(self):
        # In production the global .pth file has already called bootstrap.install().
        # Temporarily suspend that manager so this test can install its own fake
        # module patch. Restore the original manager even when an assertion fails.
        prior_manager = bootstrap.install() if bootstrap.is_installed() else None
        if prior_manager is not None:
            bootstrap.uninstall()

        name = "hcu_areal_test_target_module"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pathlib.Path(tmp, name + ".py").write_text(
                    "class Engine:\n    def run(self): return 3\n", encoding="utf-8"
                )
                sys.path.insert(0, tmp)
                importlib.invalidate_caches()
                mgr = AReaLPatchesManager()
                mgr.register_patch(
                    name + ".Engine.run",
                    lambda original: (lambda self: original(self) * 2),
                    module_name=name,
                    patch_id="test.lazy",
                    apply_wrapper=True,
                )
                try:
                    bootstrap.install(mgr)
                    module = importlib.import_module(name)
                    self.assertEqual(module.Engine().run(), 6)
                    bootstrap.uninstall()
                    self.assertEqual(module.Engine().run(), 3)
                finally:
                    bootstrap.uninstall()
                    sys.modules.pop(name, None)
                    sys.path.remove(tmp)
        finally:
            if prior_manager is not None:
                bootstrap.install(prior_manager)

    def test_lazy_import_with_global_hook_active(self):
        # Regression: exercise the exact situation when the site-packages .pth
        # has already enabled the default AWEX manager at Python startup.
        prior_manager = bootstrap.install() if bootstrap.is_installed() else None
        if prior_manager is not None:
            bootstrap.uninstall()
        try:
            bootstrap.install_if_enabled()
            self.test_lazy_import_patch_on_new_module()
            self.assertTrue(bootstrap.is_installed())
        finally:
            bootstrap.uninstall()
            if prior_manager is not None:
                bootstrap.install(prior_manager)

    def test_default_registered_but_can_disable(self):
        from hcu_areal.areal_adaptor import patch_features

        mgr = patch_features(env={})
        self.assertEqual(
            set(mgr.patch_ids),
            {
                "qwen35.awex.train.meta",
                "qwen35.awex.train.params",
                "qwen35.awex.infer.unfuse",
                "qwen35.awex.infer.sharding",
                "qwen35.awex.infer.meta",
            },
        )
        self.assertEqual(
            patch_features(env={"HCU_AREAL_DISABLE_QWEN35_AWEX": "1"}).patch_ids,
            (),
        )
