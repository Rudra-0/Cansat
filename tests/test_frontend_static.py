"""
Static checks for the web UI: the dashboard script runs inside an IIFE, so every
function referenced from an inline on*="" handler must be exported on window.
"""

import os
import re
import unittest

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
HANDLER_RE = re.compile(r'\son(?:click|change|input|submit)="\s*(?:return\s+)?([A-Za-z_]\w*)\s*\(')
EXPORT_RE = re.compile(r'window\.(\w+)\s*=')
TOP_LEVEL_FN_RE = re.compile(r'^function\s+(\w+)\s*\(', re.M)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestInlineHandlers(unittest.TestCase):

    def test_dashboard_handlers_are_exported(self):
        dashboard = _read(os.path.join(FRONTEND_DIR, "dashboard.html"))
        scripts = dashboard + "".join(
            _read(os.path.join(FRONTEND_DIR, "js", f))
            for f in os.listdir(os.path.join(FRONTEND_DIR, "js")) if f.endswith(".js"))
        handlers = set(HANDLER_RE.findall(dashboard))
        defined = set(EXPORT_RE.findall(scripts)) | set(TOP_LEVEL_FN_RE.findall(scripts))
        self.assertTrue(handlers, "expected inline handlers in dashboard.html")
        self.assertEqual(sorted(handlers - defined), [])


if __name__ == "__main__":
    unittest.main()
