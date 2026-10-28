import types
import unittest
from hcu_areal.patch_utils import AReaLPatchesManager

class TestPatchManager(unittest.TestCase):
    def test_wrap_once_and_restore(self):
        target=types.ModuleType("fake_mod")
        class Engine:
            def run(self): return "orig"
        target.Engine = Engine
        mgr=AReaLPatchesManager()
        mgr.register_patch("fake_mod.Engine.run", lambda old: (lambda self: "patched:"+old(self)),
                           module_name="fake_mod", patch_id="test.run", apply_wrapper=True)
        self.assertEqual(mgr.apply_module(target), 1)
        self.assertEqual(target.Engine().run(), "patched:orig")
        self.assertEqual(mgr.apply_module(target), 0)
        mgr.remove_patches()
        self.assertEqual(target.Engine().run(), "orig")

    def test_plain_function_and_staticmethod(self):
        target=types.ModuleType("fake_static")
        target.fn=lambda: "before"
        class Engine:
            @staticmethod
            def value(): return 3
        target.Engine=Engine
        mgr=AReaLPatchesManager()
        mgr.register_patch("fake_static.fn", lambda: "after", module_name="fake_static", patch_id="t.fn")
        mgr.register_patch("fake_static.Engine.value", lambda old: (lambda: old()+4),
                           module_name="fake_static", patch_id="t.static", apply_wrapper=True)
        mgr.apply_module(target)
        self.assertEqual(target.fn(), "after")
        self.assertEqual(Engine.value(), 7)
        mgr.remove_patches()
        self.assertEqual(target.fn(), "before")
        self.assertEqual(Engine.value(), 3)

    def test_conflicting_registration_fails(self):
        mgr=AReaLPatchesManager()
        mgr.register_patch("m.Engine.go", lambda: None, module_name="m", patch_id="a")
        with self.assertRaises(ValueError):
            mgr.register_patch("m.Engine.go", lambda: None, module_name="m", patch_id="b")
