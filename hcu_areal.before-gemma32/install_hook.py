"""Install or remove a single opt-in .pth startup hook, without editing vendor code."""
from __future__ import annotations
import argparse
import sysconfig
from pathlib import Path

NAME = "zz_hcu_areal_runtime.pth"
ROOT = Path(__file__).resolve().parent.parent
OLD_NAME = "zz_areal_runtime_patches.pth"

def render_pth(root=ROOT):
    return (f"{root}\n"
            "import hcu_areal.bootstrap as _hcu_areal_runtime; "
            "_hcu_areal_runtime.install_if_enabled()\n")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("install", "uninstall", "status"))
    ns = ap.parse_args()
    target = Path(sysconfig.get_paths()["purelib"]) / NAME
    old = target.with_name(OLD_NAME)
    content = render_pth()
    print("Python site-packages:", target.parent)
    if old.exists():
        print("WARNING: old areal_patches .pth still exists:", old)
        print("         Uninstall it using the old package before activating hcu_areal.")
    if ns.action == "status":
        print("Hook:", target)
        print("Installed:", target.exists() and target.read_text() == content)
        if target.exists() and target.read_text() != content:
            print("WARNING: hook exists with different contents")
        return
    if ns.action == "install":
        if target.exists() and target.read_text() != content:
            raise RuntimeError("Refusing to overwrite modified/unrelated file: " + str(target))
        target.write_text(content, encoding="utf-8")
        print("INSTALLED:", target)
        print("Restart Ray/Python worker processes after enabling the feature flag.")
    elif ns.action == "uninstall":
        if not target.exists():
            print("Nothing to uninstall")
        elif target.read_text() == content:
            target.unlink()
            print("REMOVED:", target)
        else:
            raise RuntimeError("Refusing to delete modified/unrelated file: " + str(target))

if __name__ == "__main__":
    main()
