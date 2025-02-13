import os
from pathlib import Path
import tempfile
import shutil
from tree_sitter import Language, Parser, Node
import subprocess
from git import Repo, Git
from git.exc import GitCommandError
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from urllib.parse import urlparse

@dataclass
class FunctionInfo:
    """Information about a Python function or method"""
    name: str
    line_number: int
    parameters: List[str]
    decorators: List[str]
    docstring: Optional[str]
    is_async: bool
    parent_class: Optional[str] = None

@dataclass
class ClassInfo:
    """Information about a Python class"""
    name: str
    line_number: int
    methods: List[FunctionInfo]
    base_classes: List[str]
    decorators: List[str]
    docstring: Optional[str]

class ModernPythonAnalyzer:
    """Analyzes Python code structure in a repository using tree-sitter"""

    def __init__(self, repo_path: str):
        """Initialize the analyzer with a repository path"""
        self.repo_path = Path(repo_path)
        self.parser = Parser()
        self.language = self._setup_tree_sitter()
        self.parser.set_language(self.language)

        # Storage for analysis results
        self.classes: Dict[str, ClassInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}

        # Define queries for code pattern matching
        self._setup_queries(self.language)

    def _setup_tree_sitter(self) -> Language:
        """Set up and return the tree-sitter Python language"""
        grammar_dir = Path('vendor/tree-sitter-python')

        # Clone grammar if missing
        if not grammar_dir.exists():
            from git import Repo
            Repo.clone_from(
                'https://github.com/tree-sitter/tree-sitter-python',
                grammar_dir
            )

        Language.build_library(
            'build/python.so',
            [str(grammar_dir)]
        )
        return Language('build/python.so', 'python')

    def _setup_queries(self, language: Language):
        """Set up tree-sitter queries for code pattern matching"""
        # Updated function query
        self.function_query = language.query("""
            (function_definition
              name: (identifier) @function.name
              parameters: (parameters) @function.params
              [decorator: (decorator) @function.decorator]*
              body: (block . (expression_statement (string) @function.docstring)?)
            ) @function.def
        """)

        # Updated class query
        self.class_query = language.query("""
            (class_definition
              name: (identifier) @class.name
              [decorator: (decorator) @class.decorator]*
              body: (block . (expression_statement (string) @class.docstring)?)
              [base_classes: (argument_list) @class.bases]?
            ) @class.def
        """)

    def _get_node_text(self, node: Node, source: bytes) -> str:
        """Extract text from a syntax tree node"""
        return source[node.start_byte:node.end_byte].decode('utf-8')

    def _extract_function_info(self, captures: List, source: bytes, parent_class: Optional[str] = None) -> FunctionInfo:
        """Extract information about a function from query captures"""
        function_node = name = docstring = None
        params = []
        decorators = []

        for node, capture_type in captures:
            if capture_type == 'function.def':
                function_node = node
            elif capture_type == 'function.name':
                name = self._get_node_text(node, source)
            elif capture_type == 'function.params':
                for param in node.named_children:
                    if param.type == 'identifier':
                        params.append(self._get_node_text(param, source))
            elif capture_type == 'function.decorator':
                decorators.append(self._get_node_text(node, source))
            elif capture_type == 'function.docstring':
                docstring = self._get_node_text(node, source)

        return FunctionInfo(
            name=name,
            line_number=function_node.start_point[0] + 1 if function_node else 0,
            parameters=params,
            decorators=decorators,
            docstring=docstring,
            is_async='async' in self._get_node_text(function_node.children[0], source) if function_node else False,
            parent_class=parent_class
        )

    def _extract_class_info(self, captures: List, source: bytes) -> ClassInfo:
        """Extract information about a class from query captures"""
        class_node = name = docstring = None
        base_classes = []
        decorators = []
        methods = []

        for node, capture_type in captures:
            if capture_type == 'class.def':
                class_node = node
            elif capture_type == 'class.name':
                name = self._get_node_text(node, source)
            elif capture_type == 'class.decorator':
                decorators.append(self._get_node_text(node, source))
            elif capture_type == 'class.docstring':
                docstring = self._get_node_text(node, source)
            elif capture_type == 'class.bases':
                for base in node.named_children:
                    base_classes.append(self._get_node_text(base, source))

        # Extract methods
        if class_node:
            for child in class_node.named_children:
                if child.type == 'block':
                    for method_node in child.named_children:
                        if method_node.type == 'function_definition':
                            method_captures = self.function_query.captures(method_node)
                            method_info = self._extract_function_info(method_captures, source, parent_class=name)
                            if method_info:
                                methods.append(method_info)

        return ClassInfo(
            name=name,
            line_number=class_node.start_point[0] + 1 if class_node else 0,
            methods=methods,
            base_classes=base_classes,
            decorators=decorators,
            docstring=docstring
        )

    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file"""
        with open(file_path, 'rb') as f:
            source = f.read()

        tree = self.parser.parse(source)

        classes = []
        functions = []

        # Analyze classes
        for match in self.class_query.matches(tree.root_node):
            class_info = self._extract_class_info(match, source)
            if class_info:
                classes.append(class_info)
                self.classes[class_info.name] = class_info

        # Analyze standalone functions
        for match in self.function_query.matches(tree.root_node):
            if match[0][0].parent.type == 'module':  # Only top-level functions
                func_info = self._extract_function_info(match, source)
                if func_info:
                    functions.append(func_info)
                    self.functions[func_info.name] = func_info

        return {
            'classes': classes,
            'functions': functions,
            'loc': len(source.splitlines())
        }

    def analyze_repo(self) -> Dict:
        """Analyze all Python files in the repository"""
        results = {}

        for py_file in self.repo_path.rglob('*.py'):
            if '.git' not in str(py_file):
                rel_path = py_file.relative_to(self.repo_path)
                file_results = self.analyze_file(py_file)
                if file_results:
                    results[str(rel_path)] = file_results

        return {
            'summary': {
                'total_files': len(results),
                'total_classes': len(self.classes),
                'total_functions': len(self.functions),
                'files_analyzed': list(results.keys())
            },
            'files': results
        }

    def generate_report(self, output_file: str = 'repo_analysis.json'):
        """Generate a JSON report of the analysis"""
        analysis_results = self.analyze_repo()

        def convert_to_dict(obj):
            if isinstance(obj, (ClassInfo, FunctionInfo)):
                return obj.__dict__
            elif isinstance(obj, (tuple, list)):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        serializable_results = convert_to_dict(analysis_results)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

def main():
    """Command line interface for the analyzer"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze Python code structure in a repository'
    )
    parser.add_argument(
        'repo_path',
        help='Path to the Python repository to analyze'
    )
    parser.add_argument(
        '--output', '-o',
        default='repo_analysis.json',
        help='Output JSON file path (default: repo_analysis.json)'
    )

    args = parser.parse_args()

    try:
        analyzer = ModernPythonAnalyzer(args.repo_path)
        analyzer.generate_report(args.output)
        print(f"Analysis complete. Results saved to {args.output}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
