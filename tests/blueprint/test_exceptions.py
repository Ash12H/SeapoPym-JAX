"""Tests for exception formatting."""

from seapopym.blueprint import DimensionMismatchError, SignatureMismatchError


class TestSignatureMismatchError:
    """Tests for SignatureMismatchError message formatting."""

    def test_with_extra(self):
        """extra= produces 'unexpected arguments' in message."""
        err = SignatureMismatchError("test:func", extra=["arg"])
        assert "unexpected arguments" in str(err)

    def test_with_both(self):
        """missing + extra → both appear in message."""
        err = SignatureMismatchError("test:func", missing=["a"], extra=["b"])
        msg = str(err)
        assert "missing arguments" in msg
        assert "unexpected arguments" in msg


class TestDimensionMismatchError:
    """Tests for DimensionMismatchError message formatting."""

    def test_with_actual(self):
        """When actual is given, 'got' appears in message."""
        err = DimensionMismatchError("state.x", expected=["Y", "X"], actual=["Z"])
        assert "got" in str(err)

    def test_without_actual(self):
        """When actual is omitted, 'got' does NOT appear."""
        err = DimensionMismatchError("state.x", expected=["Y", "X"])
        assert "got" not in str(err)
