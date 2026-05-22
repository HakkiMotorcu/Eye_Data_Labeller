"""Persist a local SAM-HeLa checkpoint path into QSettings.

Used by the install scripts so collaborators can answer the "where is
best.pt?" prompt once and have it remembered across launches.

    python configure_model.py /absolute/path/to/best.pt
    python configure_model.py --clear

Exits non-zero if the path doesn't exist (and isn't ``--clear``), so
the install script can detect a typo and re-prompt.
"""

import os
import sys


def _qsettings():
    # Use QCoreApplication so we don't need a full QApplication — this
    # script runs from install_*.command/.bat where there's no event
    # loop. The org / app names must match what main.py sets.
    from PyQt6.QtCore import QCoreApplication, QSettings
    QCoreApplication.setOrganizationName("EyeDataLabeller")
    QCoreApplication.setApplicationName("Eye Data Labeller")
    return QSettings()


def main(argv):
    if len(argv) != 2:
        print(f"usage: {os.path.basename(argv[0])} <path-to-best.pt>|--clear")
        return 2

    arg = argv[1]
    s = _qsettings()
    key = "model/sam_hela_local_path"

    if arg == "--clear":
        s.remove(key)
        s.sync()
        print("[configure_model] cleared local sam_hela path")
        return 0

    path = os.path.abspath(os.path.expanduser(arg))
    if not os.path.exists(path):
        print(f"[configure_model] file does not exist: {path}")
        return 1
    if not os.path.isfile(path):
        print(f"[configure_model] not a regular file: {path}")
        return 1

    s.setValue(key, path)
    s.sync()
    print(f"[configure_model] sam_hela local path set to:\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
