"""Tests for utils/env_check.py's dependency auto-installer. Uses mocking so
the test suite never actually shells out to pip — it only verifies the
decision logic (skip install if importable, install if not).
"""
from unittest.mock import patch

from utils.env_check import _ensure_package, ensure_dependencies


def test_ensure_package_skips_install_when_already_importable():
    with patch("subprocess.check_call") as mock_call:
        _ensure_package("os")  # stdlib module, always importable
        mock_call.assert_not_called()


def test_ensure_package_installs_when_import_fails():
    with patch("subprocess.check_call") as mock_call:
        _ensure_package("definitely_not_a_real_package_xyz")
        mock_call.assert_called_once()
        args = mock_call.call_args[0][0]
        assert "pip" in args
        assert "install" in args
        assert "definitely_not_a_real_package_xyz" in args


def test_ensure_package_uses_import_name_override_for_the_check():
    # scikit-image installs as "skimage" — the import check must use the
    # import_name override, not the pip package name, or this always
    # (wrongly) tries to reinstall it.
    with patch("subprocess.check_call") as mock_call:
        _ensure_package("scikit-image", "os")  # "os" stands in for an importable module
        mock_call.assert_not_called()


def test_ensure_dependencies_checks_all_expected_packages():
    with patch("utils.env_check._ensure_package") as mock_ensure:
        ensure_dependencies()
        called_packages = {call.args[0] for call in mock_ensure.call_args_list}
        assert called_packages == {"medpy", "ptflops", "fvcore", "scikit-image", "nibabel"}
