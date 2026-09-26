import ast
import io
import re
import sys
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ExportTests(unittest.TestCase):
    def test_python_sources_have_no_comments_or_docstrings(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8-sig")
                compile(text, str(path), "exec")
                for token in tokenize.generate_tokens(io.StringIO(text).readline):
                    if token.type == tokenize.COMMENT:
                        self.assertEqual(token.start, (1, 0))
                        self.assertTrue(token.string.startswith("#!"))
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.assertIsNone(ast.get_docstring(node))
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                        self.assertNotIsInstance(node.value.value, str)

    def test_only_main_shell_launchers_remain(self):
        self.assertEqual(
            {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.sh")},
            {"scripts/train_llm_qwen3vl.sh", "scripts/eval_qwen3vl.sh"},
        )
        self.assertFalse(list(ROOT.rglob("*.ps1")))
        self.assertFalse(list(ROOT.rglob("*.cmd")))

    def test_non_python_sources_have_no_comments(self):
        c_token = re.compile(
            r'R"([^ ()\\\t\r\n]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\])*"|'
            r"'(?:\\.|[^'\\])*'|/\*.*?\*/|//[^\n]*", re.S
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.name.lower().startswith("readme"):
                continue
            text = None
            if path.suffix in {".cpp", ".cu", ".cuh", ".h"}:
                text = path.read_text(encoding="utf-8-sig")
                for token in c_token.finditer(text):
                    self.assertFalse(token.group().startswith(("//", "/*")), path)
            elif path.suffix in {".sh", ".yaml", ".yml", ".txt"} or path.name == ".gitignore":
                text = path.read_text(encoding="utf-8-sig")
                for index, line in enumerate(text.splitlines()):
                    if index == 0 and line.startswith("#!"):
                        continue
                    self.assertNotRegex(line, r"(?:^|\s)#", path)

    def test_training_does_not_print_adapter_configuration(self):
        path = ROOT / "src/georisk/llm/train_uav_qwen.py"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("print_trainable_parameters", text)
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "logger":
                continue
            args = ast.dump(ast.Tuple(elts=node.args, ctx=ast.Load()))
            self.assertFalse(any(isinstance(arg, ast.Name) and arg.id == "model_args" for arg in node.args))
            self.assertNotRegex(args, r"requested_visual_blocks|target_modules|visual_targets|merger_targets|visual_lora_block_indices|visual_lora_last_n")
            if node.func.attr == "info":
                self.assertNotRegex(args.lower(), r"lora|trainable params")
        shell = (ROOT / "scripts/train_llm_qwen3vl.sh").read_text(encoding="utf-8")
        for line in shell.splitlines():
            if re.match(r"\s*(echo|printf)\b", line):
                self.assertNotRegex(line.lower(), r"visual.?lora|lora.?r.?alpha|block_indices|last_n|cmd\[@\]")

    def test_sources_have_no_machine_specific_paths(self):
        pattern = re.compile(r"/(?:root|home|data\d+)/|\b[A-Z]:[\\/]", re.I)
        extensions = {".py", ".sh", ".ps1", ".cmd", ".yaml", ".md", ".txt"}
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertIsNone(pattern.search(path.read_text(encoding="utf-8-sig")))

    def test_navigation_view_contract(self):
        from georisk.model_wrapper.utils.travel_qwen_util import _resolve_view_mode_indices
        self.assertEqual(_resolve_view_mode_indices("dual"), [0, 4])
        for mode in ("single", "five"):
            with self.assertRaises(ValueError):
                _resolve_view_mode_indices(mode)


if __name__ == "__main__":
    unittest.main()
