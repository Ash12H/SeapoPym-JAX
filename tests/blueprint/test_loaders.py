"""Tests for Blueprint file loaders (load_json, load_yaml, load_file)."""

import json

import pytest
import yaml

from seapopym.blueprint.loaders import load_file, load_json, load_yaml


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


class TestLoadJson:
    """Tests for load_json."""

    def test_valid_json(self, tmp_dir):
        """Valid JSON file returns a dict."""
        path = tmp_dir / "data.json"
        path.write_text(json.dumps({"key": "value"}))
        result = load_json(path)
        assert result == {"key": "value"}

    def test_non_dict_root(self, tmp_dir):
        """JSON with non-dict root raises ValueError."""
        path = tmp_dir / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="Expected a dictionary at root level"):
            load_json(path)


class TestLoadYaml:
    """Tests for load_yaml."""

    def test_empty_yaml(self, tmp_dir):
        """Empty YAML file returns empty dict."""
        path = tmp_dir / "empty.yaml"
        path.write_text("")
        result = load_yaml(path)
        assert result == {}

    def test_non_dict_root(self, tmp_dir):
        """YAML with list root raises ValueError."""
        path = tmp_dir / "list.yaml"
        path.write_text(yaml.dump([1, 2, 3]))
        with pytest.raises(ValueError, match="Expected a dictionary at root level"):
            load_yaml(path)


class TestLoadFile:
    """Tests for load_file auto-detection."""

    def test_json_extension(self, tmp_dir):
        """load_file routes .json files to load_json."""
        path = tmp_dir / "x.json"
        path.write_text(json.dumps({"format": "json"}))
        result = load_file(path)
        assert result == {"format": "json"}

    def test_yaml_extension(self, tmp_dir):
        """load_file routes .yaml files to load_yaml."""
        path = tmp_dir / "x.yaml"
        path.write_text(yaml.dump({"format": "yaml"}))
        result = load_file(path)
        assert result == {"format": "yaml"}

    def test_yml_extension(self, tmp_dir):
        """load_file routes .yml files to load_yaml."""
        path = tmp_dir / "x.yml"
        path.write_text(yaml.dump({"format": "yml"}))
        result = load_file(path)
        assert result == {"format": "yml"}
