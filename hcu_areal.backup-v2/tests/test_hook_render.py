from pathlib import Path
import unittest
from hcu_areal.install_hook import render_pth

class TestHookFile(unittest.TestCase):
    def test_name_and_import(self):
        s=render_pth(Path("/tmp/a_repo"))
        self.assertIn("/tmp/a_repo\n",s)
        self.assertIn("import hcu_areal.bootstrap",s)
        self.assertIn("install_if_enabled()",s)
