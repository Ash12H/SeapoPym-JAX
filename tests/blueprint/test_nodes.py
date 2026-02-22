"""Tests for DataNode and ComputeNode identity (eq/hash)."""

from seapopym.blueprint import ComputeNode, DataNode


class TestDataNode:
    """Tests for DataNode __eq__ and __hash__."""

    def test_eq_same_name(self):
        """Two DataNodes with the same name are equal, regardless of other fields."""
        a = DataNode(name="state.biomass", dims=("Y", "X"), units="g")
        b = DataNode(name="state.biomass", dims=("Z",), units="kg")
        assert a == b

    def test_eq_different_name(self):
        """DataNodes with different names are not equal."""
        a = DataNode(name="state.biomass")
        b = DataNode(name="state.temperature")
        assert a != b

    def test_eq_not_implemented(self):
        """Comparison with a non-DataNode returns NotImplemented."""
        node = DataNode(name="state.biomass")
        assert node.__eq__("not a node") is NotImplemented

    def test_hash_consistency(self):
        """Same name → same hash; usable in set."""
        a = DataNode(name="state.biomass", dims=("Y", "X"))
        b = DataNode(name="state.biomass", dims=("Z",))
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


class TestComputeNode:
    """Tests for ComputeNode __eq__ and __hash__."""

    def test_eq_same_name(self):
        """Two ComputeNodes with the same name are equal."""
        a = ComputeNode(func=lambda x: x, name="step1")
        b = ComputeNode(func=lambda x: x * 2, name="step1")
        assert a == b

    def test_eq_not_implemented(self):
        """Comparison with a non-ComputeNode returns NotImplemented."""
        node = ComputeNode(func=lambda x: x, name="step1")
        assert node.__eq__("not a node") is NotImplemented

    def test_hash_consistency(self):
        """Same name → same hash; usable in set."""
        a = ComputeNode(func=lambda x: x, name="step1")
        b = ComputeNode(func=lambda x: x * 2, name="step1")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
