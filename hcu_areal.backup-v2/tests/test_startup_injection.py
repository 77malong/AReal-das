"""Simulate separate train/inference worker interpreters with a .pth startup hook."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from hcu_areal.install_hook import render_pth, ROOT


class TestStartupInChild(unittest.TestCase):
    def test_separate_workers_inherit_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake = tmp / "stub_source"
            train = fake / "areal" / "v2" / "weight_update" / "awex"
            train.mkdir(parents=True)
            for path in [fake / "areal", fake / "areal" / "v2",
                         fake / "areal" / "v2" / "weight_update", train]:
                (path / "__init__.py").write_text("", encoding="utf-8")
            (train / "fsdp_adapter.py").write_text(
                'from types import SimpleNamespace\n'
                'class AwexFSDPAdapter:\n'
                '    def __init__(self):\n'
                '        self._engine=SimpleNamespace(model_config=SimpleNamespace(model_type="qwen3_5"))\n'
                '    def get_weight_metadata(self):\n'
                '        return []\n', encoding="utf-8")
            (train / "sglang_adapter.py").write_text(
                'from types import SimpleNamespace\n'
                'class AwexSGLangAdapter:\n'
                '    def _get_model(self):\n'
                '        return SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))\n'
                '    def get_weight_metadata(self):\n'
                '        return []\n', encoding="utf-8")
            hooksite = tmp / "hooksite"
            hooksite.mkdir()
            (hooksite / "zz_hcu_areal_runtime.pth").write_text(render_pth(ROOT), encoding="utf-8")
            trace = tmp / "trace"
            code = (
                'import site,sys\n'
                f'sys.path.insert(0,{str(fake)!r})\n'
                f'site.addsitedir({str(hooksite)!r})\n'
                'from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter\n'
                'from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter\n'
                'a=AwexFSDPAdapter();b=AwexSGLangAdapter()\n'
                'assert a.get_weight_metadata()==[]\n'
                'assert b.get_weight_metadata()==[]\n'
                'assert "qwen35.awex.trace.train" in AwexFSDPAdapter.get_weight_metadata.__hcu_areal_patch_ids__\n'
                'assert "qwen35.awex.trace.infer" in AwexSGLangAdapter.get_weight_metadata.__hcu_areal_patch_ids__\n'
                'print("PASS child pid", __import__("os").getpid())\n'
            )
            env = dict(os.environ, HCU_AREAL_PATCH_QWEN35_AWEX="1",
                       HCU_AREAL_PATCH_TRACE_DIR=str(trace))
            for worker in ("train", "infer"):
                process = subprocess.run(
                    [sys.executable, "-S", "-c", code], env=env,
                    text=True, capture_output=True, cwd=str(tmp), timeout=30,
                )
                self.assertEqual(process.returncode, 0,
                                 f"{worker}\nstdout={process.stdout}\nstderr={process.stderr}")
                self.assertIn("PASS child pid", process.stdout)
            self.assertEqual(len(list(trace.glob("*.json"))), 4)

if __name__ == '__main__':
    unittest.main()
