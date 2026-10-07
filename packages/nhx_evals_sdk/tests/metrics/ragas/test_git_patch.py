# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test module to validate the git patch functionality.
This can be used to verify that the patch works correctly.
"""

import sys
import unittest

from nhx_evals_sdk.metrics.ragas.git_patch import apply_git_patch


class TestGitPatch(unittest.TestCase):
    """Test cases for the git patch functionality."""

    def setUp(self):
        """Clean up git module from sys.modules before each test."""
        # Remove git modules if they exist
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith("git")]
        for module in modules_to_remove:
            del sys.modules[module]

    def tearDown(self):
        """Clean up git module from sys.modules after each test."""
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith("git")]
        for module in modules_to_remove:
            del sys.modules[module]

    def test_apply_git_patch_creates_mock_module(self):
        """Test that apply_git_patch creates a mock git module."""
        # Ensure git is not imported
        self.assertNotIn("git", sys.modules)

        # Apply the patch
        apply_git_patch()

        # Verify git module is now available
        self.assertIn("git", sys.modules)

        # Verify the refresh function is available and is a no-op
        import git

        assert git.__name__ == "git"

    def test_ragas_import_after_patch(self):
        """Test that ragas can be imported after applying the patch."""
        apply_git_patch()

        # This should not raise an exception
        try:
            import ragas as ragas
        except:  # noqa
            self.fail("Unexpected exception after applying git patch: ImportError")


if __name__ == "__main__":
    unittest.main()
