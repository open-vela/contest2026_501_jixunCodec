#!/usr/bin/env python3
"""Integration checks for role selection, asset identity and output protection."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from materialize_aibox import materialize, PROFILE, REPO


class RoleSelectionTest(unittest.TestCase):
    def test_role_and_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            tx = materialize("tx", Path(directory) / "tx")
            rx = materialize("rx", Path(directory) / "rx")
            self.assertIn("#define JX_SIMPLE_RX 0", (tx / "port/jx_simple_role.h").read_text())
            self.assertIn("#define JX_SIMPLE_RX 1", (rx / "port/jx_simple_role.h").read_text())
            self.assertIn("jx_ui_receiver_autostart", (rx / "main/jx_main.c").read_text(encoding="utf-8"))
            self.assertIn("atoi(lead) : 100", (tx / "deps/audio/ai_module/ai_voice/ai_voice.c").read_text(encoding="utf-8"))
            for weight in (PROFILE / "common/weights/3k").rglob("*"):
                if weight.is_file():
                    rel = weight.relative_to(PROFILE / "common")
                    self.assertEqual(hashlib.sha256((tx / rel).read_bytes()).digest(), hashlib.sha256((rx / rel).read_bytes()).digest())

    def test_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "application"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError): materialize("tx", target)
            self.assertEqual(marker.read_text(), "keep")

    def test_source_tree_and_invalid_role_rejected(self):
        with self.assertRaises(ValueError): materialize("tx", REPO / "new-output")
        with self.assertRaises(ValueError): materialize("other", Path(tempfile.gettempdir()) / "invalid-role")


if __name__ == "__main__": unittest.main()
