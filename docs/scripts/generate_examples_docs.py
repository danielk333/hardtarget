import pathlib
import os

try:
    root = pathlib.Path(__file__).parent.parent.parent
except NameError:
    root = pathlib.Path(os.path.abspath(""))
docs = root / "docs"
examples = docs / "examples"
notebooks = docs / "notebooks"

base = ["  - Examples:", "    - examples/index.md"]


def add_tree(base, folder):
    for file in sorted(folder.rglob("*.py")):
        example_path = file.relative_to(docs)

        if file.parent.name.startswith("."):
            continue
        if file.name.startswith("_wip"):
            continue

        base.append(f"    - {file.stem}: {str(example_path)}")


add_tree(base, examples)
base.append("  - Notebooks:")
base.append("    - notebooks/index.md")
add_tree(base, notebooks)

print("\n".join(base))
