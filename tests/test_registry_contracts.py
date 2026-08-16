import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"VipeEstimate", "VipeSaveData", "VipeLoadData", "VipeExtractData", "VipeSelectMasks"}


def mapping_keys(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError(f"{name} not found")


class RegistryContractTests(unittest.TestCase):
    def test_registry_is_complete_and_symmetric(self):
        tree = ast.parse((ROOT / "nodes" / "nodes.py").read_text())
        self.assertEqual(mapping_keys(tree, "NODE_CLASS_MAPPINGS"), EXPECTED)
        self.assertEqual(mapping_keys(tree, "NODE_DISPLAY_NAME_MAPPINGS"), EXPECTED)

    def test_all_nodes_have_comfy_contracts(self):
        tree = ast.parse((ROOT / "nodes" / "nodes.py").read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.startswith("Vipe")]
        self.assertEqual({node.name for node in classes}, EXPECTED)
        for node in classes:
            fields = {
                target.id
                for statement in node.body
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
            methods = {statement.name for statement in node.body if isinstance(statement, ast.FunctionDef)}
            self.assertTrue({"INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"} <= fields | methods)

    def test_loader_is_input_only(self):
        source = (ROOT / "utils" / "paths.py").read_text()
        self.assertIn("get_input_directory", source)
        self.assertIn("resolve(strict=True)", source)
        self.assertIn("is_relative_to(root)", source)

    def test_vipe_data_producers_use_lowercase_output_label(self):
        tree = ast.parse((ROOT / "nodes" / "nodes.py").read_text())
        producers = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in {"VipeEstimate", "VipeLoadData"}
        }
        for name, node in producers.items():
            return_names = next(
                statement.value
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "RETURN_NAMES" for target in statement.targets)
            )
            self.assertEqual(ast.literal_eval(return_names), ("vipe_data",), name)


if __name__ == "__main__":
    unittest.main()
