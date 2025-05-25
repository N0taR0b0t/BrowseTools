import os
import re
import sys
from graphviz import Digraph

visited_files = set()
file_nodes = {}
dependency_graph = Digraph(format='png')
base_directory = None

def extract_from_imports(file_path):
    """Extract 'from module import something' lines from a file."""
    from_imports = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("from ") and "import" in line:
                parts = line.split()
                if len(parts) >= 4:
                    module = parts[1]
                    from_imports.append(module)
    return from_imports

def resolve_module_path(module_name):
    """Try to resolve module name to a .py file in the base_path."""
    module_path = module_name.replace('.', os.sep) + '.py'
    potential_path = os.path.join(base_directory, module_path)
    if os.path.isfile(potential_path):
        return os.path.abspath(potential_path)
    return None

def label_for_path(file_path):
    """Return a shorter label for display in the graph."""
    return os.path.relpath(file_path, base_directory)

def collect_dependencies(file_path):
    """Recursively collect all dependencies via 'from' imports."""
    file_path = os.path.abspath(file_path)
    if file_path in visited_files:
        return
    visited_files.add(file_path)

    print(f"Scanning: {label_for_path(file_path)}")
    file_label = label_for_path(file_path)
    dependency_graph.node(file_label)

    imports = extract_from_imports(file_path)
    for module in imports:
        dep_path = resolve_module_path(module)
        if dep_path:
            dep_label = label_for_path(dep_path)
            dependency_graph.edge(file_label, dep_label)
            collect_dependencies(dep_path)
        else:
            print(f"  Skipped or not found: {module}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python viewFlow.py <path_to_py_file>")
        sys.exit(1)

    entry_file = sys.argv[1]
    base_directory = os.path.dirname(os.path.abspath(entry_file))

    collect_dependencies(entry_file)

    output_path = os.path.splitext(os.path.basename(entry_file))[0] + "_dep_graph"
    dependency_graph.render(output_path, view=True)
    print(f"\nFlowchart generated: {output_path}.png")