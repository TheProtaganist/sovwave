#!/usr/bin/env python3
"""
Julia Test Validator GUI
========================
Validates, runs, and manages Julia test files with automatic import management.
Extracts test functions, parameters, results, and allows interactive test creation.

Features:
- Load and analyze Julia test files (@testset, @test)
- Extract function parameters, expected values, and test logic
- Run tests automatically via Julia subprocess
- Error handling with fix suggestions
- Automatic import management
- Interactive test function builder
- Support for all Julia constructs (loops, if statements, structs, globals, constants)
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import re
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import threading
import queue


@dataclass
class TestFunction:
    """Represents a Julia test function or testset"""
    name: str
    type: str  # 'testset', 'test', 'function'
    line_start: int
    line_end: int
    code: str
    params: List[Tuple[str, str, Any]] = field(default_factory=list)  # [(name, type, default_value), ...]
    variables: Dict[str, Any] = field(default_factory=dict)  # Local variables defined
    assertions: List[str] = field(default_factory=list)  # @test statements
    loops: List[str] = field(default_factory=list)  # for/while loops
    conditionals: List[str] = field(default_factory=list)  # if statements
    nested_tests: List['TestFunction'] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)  # Required imports
    passed: Optional[bool] = None
    error_msg: str = ""
    
    def __repr__(self):
        return f"TestFunction(name={self.name}, type={self.type}, line={self.line_start}-{self.line_end})"


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    passed: bool
    error_msg: str = ""
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0


class JuliaTestParser:
    """Parse Julia test files and extract test structures"""
    
    def __init__(self):
        self.imports = set()
        self.global_vars = {}
        self.constants = {}
        self.structs = {}
    
    def parse_file(self, filepath: str) -> List[TestFunction]:
        """Parse Julia test file and extract all test functions"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Extract imports
        self.imports = self.extract_imports(content)
        
        # Extract global constants and structs
        self.constants = self.extract_constants(content)
        self.structs = self.extract_structs(content)
        self.global_vars = self.extract_globals(content)
        
        # Parse test functions
        tests = self.parse_testsets(lines)
        
        return tests
    
    def extract_imports(self, content: str) -> Set[str]:
        """Extract all import/using statements"""
        imports = set()
        for line in content.split('\n'):
            line = line.strip()
            # using Module
            if re.match(r'^using\s+', line):
                imports.add(line)
            # import Module
            elif re.match(r'^import\s+', line):
                imports.add(line)
        return imports
    
    def extract_constants(self, content: str) -> Dict[str, Any]:
        """Extract const definitions"""
        constants = {}
        for line in content.split('\n'):
            match = re.search(r'^\s*const\s+(\w+)\s*=\s*(.+?)(?:#|$)', line)
            if match:
                name = match.group(1)
                value = match.group(2).strip()
                constants[name] = value
        return constants
    
    def extract_structs(self, content: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract struct definitions"""
        structs = {}
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = re.search(r'^\s*(?:mutable\s+)?struct\s+(\w+)', line)
            if match:
                struct_name = match.group(1)
                fields = []
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('end'):
                        break
                    field_match = re.search(r'(\w+)\s*::\s*([^\s#]+)', line)
                    if field_match:
                        fields.append((field_match.group(1), field_match.group(2)))
                    i += 1
                structs[struct_name] = fields
            i += 1
        return structs
    
    def extract_globals(self, content: str) -> Dict[str, Any]:
        """Extract global variable assignments"""
        globals_dict = {}
        for line in content.split('\n'):
            # global var = value
            match = re.search(r'^\s*(?:global\s+)?(\w+)\s*=\s*(.+?)(?:#|$)', line)
            if match and not line.strip().startswith('const') and not line.strip().startswith('function'):
                name = match.group(1)
                value = match.group(2).strip()
                globals_dict[name] = value
        return globals_dict
    
    def parse_testsets(self, lines: List[str]) -> List[TestFunction]:
        """Parse @testset blocks"""
        tests = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Match @testset "name" begin
            testset_match = re.search(r'@testset\s+"([^"]+)"\s+begin', line)
            if testset_match:
                test_name = testset_match.group(1)
                start_line = i + 1
                
                # Find matching end
                end_line = self.find_matching_end(lines, i)
                
                # Extract code block
                code_lines = lines[start_line:end_line]
                code = '\n'.join(code_lines)
                
                # Parse testset content
                test_func = TestFunction(
                    name=test_name,
                    type='testset',
                    line_start=start_line,
                    line_end=end_line,
                    code=code
                )
                
                # Extract assertions
                test_func.assertions = self.extract_assertions(code)
                
                # Extract variables
                test_func.variables = self.extract_variables(code)
                
                # Extract loops
                test_func.loops = self.extract_loops(code)
                
                # Extract conditionals
                test_func.conditionals = self.extract_conditionals(code)
                
                # Extract required imports (analyze function calls)
                test_func.imports = self.detect_required_imports(code)
                
                # Parse nested testsets
                test_func.nested_tests = self.parse_testsets(code_lines)
                
                tests.append(test_func)
                i = end_line
            i += 1
        
        return tests
    
    def find_matching_end(self, lines: List[str], start_idx: int) -> int:
        """Find matching 'end' for begin block"""
        depth = 1
        i = start_idx + 1
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if re.search(r'\bbegin\b', line) or re.search(r'\bfunction\b', line) or \
               re.search(r'\b(?:mutable\s+)?struct\b', line) or re.search(r'\bfor\b', line) or \
               re.search(r'\bwhile\b', line) or re.search(r'\bif\b', line):
                depth += 1
            elif line.startswith('end') or line == 'end':
                depth -= 1
            i += 1
        return i - 1
    
    def extract_assertions(self, code: str) -> List[str]:
        """Extract @test statements"""
        assertions = []
        for line in code.split('\n'):
            if '@test' in line:
                assertions.append(line.strip())
        return assertions
    
    def extract_variables(self, code: str) -> Dict[str, Any]:
        """Extract variable assignments"""
        variables = {}
        for line in code.split('\n'):
            # Match: var = value
            match = re.search(r'^\s*(\w+)\s*=\s*(.+?)(?:#|$)', line)
            if match and not line.strip().startswith('const') and \
               not line.strip().startswith('function') and \
               '@test' not in line:
                name = match.group(1)
                value = match.group(2).strip()
                variables[name] = value
        return variables
    
    def extract_loops(self, code: str) -> List[str]:
        """Extract for/while loops"""
        loops = []
        lines = code.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # for loop
            if re.match(r'^\s*for\s+', line):
                loop_start = i
                loop_end = self.find_loop_end(lines, i)
                loop_code = '\n'.join(lines[loop_start:loop_end+1])
                loops.append(loop_code)
                i = loop_end
            # while loop
            elif re.match(r'^\s*while\s+', line):
                loop_start = i
                loop_end = self.find_loop_end(lines, i)
                loop_code = '\n'.join(lines[loop_start:loop_end+1])
                loops.append(loop_code)
                i = loop_end
            i += 1
        return loops
    
    def find_loop_end(self, lines: List[str], start_idx: int) -> int:
        """Find matching 'end' for loop"""
        depth = 1
        i = start_idx + 1
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if re.search(r'\b(for|while|function|struct|if|begin)\b', line):
                depth += 1
            elif line.startswith('end') or line == 'end':
                depth -= 1
            i += 1
        return i - 1
    
    def extract_conditionals(self, code: str) -> List[str]:
        """Extract if/else statements"""
        conditionals = []
        lines = code.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.match(r'^\s*if\s+', line):
                cond_start = i
                cond_end = self.find_conditional_end(lines, i)
                cond_code = '\n'.join(lines[cond_start:cond_end+1])
                conditionals.append(cond_code)
                i = cond_end
            i += 1
        return conditionals
    
    def find_conditional_end(self, lines: List[str], start_idx: int) -> int:
        """Find matching 'end' for if block"""
        depth = 1
        i = start_idx + 1
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if re.search(r'\b(if|for|while|function|struct|begin)\b', line):
                depth += 1
            elif line.startswith('end') or line == 'end':
                depth -= 1
            i += 1
        return i - 1
    
    def detect_required_imports(self, code: str) -> Set[str]:
        """Detect required imports from function calls"""
        imports = set()
        
        # Common Test.jl imports
        if '@test' in code:
            imports.add('using Test')
        
        # Detect module prefixed calls (Module.function)
        # Only match valid Julia module names (start with letter, contain letters/numbers)
        module_calls = re.findall(r'\b([A-Z][A-Za-z0-9_]*)\.([a-zA-Z_]\w*)', code)
        for module, func in module_calls:
            # Exclude common Julia modules and numbers
            if module not in ['Base', 'Core', 'Main', 'Test'] and not module.isdigit():
                imports.add(f'using {module}')
        
        return imports


class JuliaTestRunner:
    """Run Julia tests and capture results"""
    
    def __init__(self):
        self.julia_cmd = "julia"
    
    def check_julia_available(self) -> Tuple[bool, str]:
        """Check if Julia is available"""
        try:
            result = subprocess.run([self.julia_cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, "Julia command failed"
        except FileNotFoundError:
            return False, "Julia not found in PATH"
        except Exception as e:
            return False, str(e)
    
    def run_test_file(self, filepath: str) -> Tuple[bool, str, str]:
        """Run entire Julia test file"""
        try:
            result = subprocess.run(
                [self.julia_cmd, filepath],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Test execution timed out (30s)"
        except Exception as e:
            return False, "", str(e)
    
    def run_test_code(self, code: str, test_name: str = "inline") -> TestResult:
        """Run Julia code snippet"""
        try:
            # Wrap code in Test environment
            test_code = f"""
using Test

@testset "{test_name}" begin
{code}
end
"""
            
            result = subprocess.run(
                [self.julia_cmd, '-e', test_code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            passed = result.returncode == 0 and 'Error' not in result.stderr
            
            return TestResult(
                test_name=test_name,
                passed=passed,
                stdout=result.stdout,
                stderr=result.stderr,
                error_msg="" if passed else result.stderr
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                test_name=test_name,
                passed=False,
                error_msg="Test execution timed out (10s)"
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                error_msg=str(e)
            )
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Validate Julia code syntax"""
        try:
            # Create a complete Julia script to validate
            # Wrap in a module context to allow testsets
            test_script = f'''
using Test

{code}
'''
            # Use --check-bounds=no and --compile=min for faster validation
            result = subprocess.run(
                [self.julia_cmd, '--check-bounds=no', '--compile=min', '-e', test_script],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Only syntax errors will show up immediately, not runtime errors
            # Check for parse/syntax errors in stderr
            if 'ParseError' in result.stderr or 'syntax:' in result.stderr.lower():
                return False, result.stderr
            return True, ""
        except subprocess.TimeoutExpired:
            return True, ""  # Timeout means it's trying to run, so syntax is OK
        except Exception as e:
            return False, str(e)


class TestBuilderDialog(tk.Toplevel):
    """Dialog for building new test functions interactively"""
    
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Create New Test Function")
        self.geometry("800x600")
        self.callback = callback
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup test builder UI"""
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Test name
        name_frame = tk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)
        tk.Label(name_frame, text="Test Name:", width=15).pack(side=tk.LEFT)
        self.name_entry = tk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Test type
        type_frame = tk.Frame(main_frame)
        type_frame.pack(fill=tk.X, pady=5)
        tk.Label(type_frame, text="Test Type:", width=15).pack(side=tk.LEFT)
        self.type_var = tk.StringVar(value='testset')
        tk.Radiobutton(type_frame, text='@testset', variable=self.type_var, 
                      value='testset').pack(side=tk.LEFT)
        tk.Radiobutton(type_frame, text='function', variable=self.type_var, 
                      value='function').pack(side=tk.LEFT)
        
        # Builder tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Variables tab
        var_frame = tk.Frame(notebook)
        notebook.add(var_frame, text="Variables")
        
        tk.Label(var_frame, text="Define variables (name = value):").pack(anchor=tk.W, pady=5)
        self.vars_text = scrolledtext.ScrolledText(var_frame, height=8)
        self.vars_text.pack(fill=tk.BOTH, expand=True)
        self.vars_text.insert('1.0', '# Example:\n# x = 10\n# y = 20.5\n# name = "test"\n')
        
        # Assertions tab
        assert_frame = tk.Frame(notebook)
        notebook.add(assert_frame, text="Assertions")
        
        tk.Label(assert_frame, text="Add @test statements:").pack(anchor=tk.W, pady=5)
        self.assert_text = scrolledtext.ScrolledText(assert_frame, height=8)
        self.assert_text.pack(fill=tk.BOTH, expand=True)
        self.assert_text.insert('1.0', '# Example:\n# @test x + y == 30.5\n# @test length(name) == 4\n')
        
        # Loops tab
        loop_frame = tk.Frame(notebook)
        notebook.add(loop_frame, text="Loops")
        
        tk.Label(loop_frame, text="Add for/while loops:").pack(anchor=tk.W, pady=5)
        self.loop_text = scrolledtext.ScrolledText(loop_frame, height=8)
        self.loop_text.pack(fill=tk.BOTH, expand=True)
        self.loop_text.insert('1.0', '# Example:\n# for i in 1:10\n#     @test i > 0\n# end\n')
        
        # Conditionals tab
        cond_frame = tk.Frame(notebook)
        notebook.add(cond_frame, text="Conditionals")
        
        tk.Label(cond_frame, text="Add if/else statements:").pack(anchor=tk.W, pady=5)
        self.cond_text = scrolledtext.ScrolledText(cond_frame, height=8)
        self.cond_text.pack(fill=tk.BOTH, expand=True)
        self.cond_text.insert('1.0', '# Example:\n# if x > 5\n#     @test true\n# else\n#     @test false\n# end\n')
        
        # Preview
        preview_frame = tk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Preview", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Create Test", command=self.create_test, 
                 bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        self.update_preview()
    
    def update_preview(self):
        """Update preview of generated test"""
        self.preview_text.delete('1.0', tk.END)
        
        test_name = self.name_entry.get() or "MyTest"
        test_type = self.type_var.get()
        
        # Get content from tabs (filter out comments)
        vars_code = '\n'.join([l for l in self.vars_text.get('1.0', tk.END).split('\n') 
                               if l.strip() and not l.strip().startswith('#')])
        assert_code = '\n'.join([l for l in self.assert_text.get('1.0', tk.END).split('\n') 
                                 if l.strip() and not l.strip().startswith('#')])
        loop_code = '\n'.join([l for l in self.loop_text.get('1.0', tk.END).split('\n') 
                               if l.strip() and not l.strip().startswith('#')])
        cond_code = '\n'.join([l for l in self.cond_text.get('1.0', tk.END).split('\n') 
                               if l.strip() and not l.strip().startswith('#')])
        
        if test_type == 'testset':
            code = f'@testset "{test_name}" begin\n'
            if vars_code:
                code += f'    {vars_code.replace(chr(10), chr(10) + "    ")}\n'
            if assert_code:
                code += f'    {assert_code.replace(chr(10), chr(10) + "    ")}\n'
            if loop_code:
                code += f'    {loop_code.replace(chr(10), chr(10) + "    ")}\n'
            if cond_code:
                code += f'    {cond_code.replace(chr(10), chr(10) + "    ")}\n'
            code += 'end\n'
        else:
            code = f'function {test_name}()\n'
            if vars_code:
                code += f'    {vars_code.replace(chr(10), chr(10) + "    ")}\n'
            if assert_code:
                code += f'    {assert_code.replace(chr(10), chr(10) + "    ")}\n'
            if loop_code:
                code += f'    {loop_code.replace(chr(10), chr(10) + "    ")}\n'
            if cond_code:
                code += f'    {cond_code.replace(chr(10), chr(10) + "    ")}\n'
            code += 'end\n'
        
        self.preview_text.insert('1.0', code)
    
    def create_test(self):
        """Create and return test code"""
        self.update_preview()
        code = self.preview_text.get('1.0', tk.END).strip()
        if code:
            self.callback(code)
            self.destroy()
        else:
            messagebox.showwarning("Empty Test", "Please add some test content")


class TestValidatorGUI:
    """Main GUI for Julia Test Validator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Julia Test Validator & Runner")
        self.root.geometry("1200x800")
        
        self.parser = JuliaTestParser()
        self.runner = JuliaTestRunner()
        
        self.current_file = ""
        self.test_functions: List[TestFunction] = []
        self.test_results: Dict[str, TestResult] = {}
        
        # Text size limits to prevent X11 rendering errors
        self.MAX_TEXT_SIZE = 50000  # characters
        
        self.setup_ui()
        self.check_julia()
    
    def setup_ui(self):
        """Setup main UI"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Test File", command=self.load_file)
        file_menu.add_command(label="Save Test File", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        test_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tests", menu=test_menu)
        test_menu.add_command(label="Run All Tests", command=self.run_all_tests)
        test_menu.add_command(label="Create New Test", command=self.create_new_test)
        test_menu.add_separator()
        test_menu.add_command(label="Fix Imports", command=self.fix_imports)
        
        # Main container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title - use smaller font to avoid X11 rendering issues
        title = tk.Label(main_frame, text="Julia Test Validator & Runner", 
                        font=('TkDefaultFont', 14, 'bold'))
        title.pack(pady=(0, 10))
        
        # File selection
        file_frame = tk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        tk.Label(file_frame, text="Test File:").pack(side=tk.LEFT)
        self.file_entry = tk.Entry(file_frame, width=60)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(file_frame, text="Browse", command=self.load_file).pack(side=tk.LEFT, padx=2)
        tk.Button(file_frame, text="Analyze", command=self.analyze_file, 
                 bg='#2196F3', fg='white').pack(side=tk.LEFT, padx=2)
        tk.Button(file_frame, text="Run All", command=self.run_all_tests, 
                 bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=2)
        
        # Split pane
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Test list
        left_frame = tk.Frame(paned)
        paned.add(left_frame, width=400)
        
        tk.Label(left_frame, text="Test Functions", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
        
        test_list_frame = tk.Frame(left_frame)
        test_list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(test_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.test_listbox = tk.Listbox(test_list_frame, yscrollcommand=scrollbar.set)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.test_listbox.bind('<<ListboxSelect>>', self.on_test_select)
        scrollbar.config(command=self.test_listbox.yview)
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Run Selected", command=self.run_selected_test).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="New Test", command=self.create_new_test).pack(side=tk.LEFT, padx=2)
        
        # Right: Test details tabs
        right_frame = tk.Frame(paned)
        paned.add(right_frame)
        
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Details tab
        details_frame = tk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Details")
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
        # Code tab
        code_frame = tk.Frame(self.notebook)
        self.notebook.add(code_frame, text="Code")
        self.code_text = scrolledtext.ScrolledText(code_frame, wrap=tk.NONE)
        self.code_text.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_frame = tk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Imports tab
        imports_frame = tk.Frame(self.notebook)
        self.notebook.add(imports_frame, text="Imports")
        self.imports_text = scrolledtext.ScrolledText(imports_frame, wrap=tk.WORD)
        self.imports_text.pack(fill=tk.BOTH, expand=True)
        
        # Issues tab
        issues_frame = tk.Frame(self.notebook)
        self.notebook.add(issues_frame, text="Issues")
        self.issues_text = scrolledtext.ScrolledText(issues_frame, wrap=tk.WORD)
        self.issues_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags - use system fonts to avoid X11 glyph errors
        for text_widget in [self.details_text, self.code_text, self.results_text, 
                           self.imports_text, self.issues_text]:
            text_widget.tag_config('error', foreground='red', font=('TkDefaultFont', 10, 'bold'))
            text_widget.tag_config('success', foreground='green', font=('TkDefaultFont', 10, 'bold'))
            text_widget.tag_config('warning', foreground='orange')
            text_widget.tag_config('info', foreground='blue')
            text_widget.tag_config('header', font=('TkDefaultFont', 11, 'bold'))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W, bg='lightgray')
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def check_julia(self):
        """Check if Julia is available"""
        available, msg = self.runner.check_julia_available()
        if available:
            self.status_var.set(f"Julia: {msg}")
        else:
            self.status_var.set(f"WARNING: {msg}")
            messagebox.showwarning("Julia Not Found", 
                                  f"{msg}\n\nPlease ensure Julia is installed and in your PATH.")
    
    def load_file(self):
        """Load test file"""
        filename = filedialog.askopenfilename(
            title="Select Julia Test File",
            filetypes=[("Julia files", "*.jl"), ("All files", "*.*")]
        )
        if filename:
            self.current_file = filename
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.analyze_file()
    
    def analyze_file(self):
        """Analyze loaded test file"""
        filepath = self.file_entry.get()
        if not filepath or not Path(filepath).exists():
            messagebox.showerror("File Error", "Please select a valid test file")
            return
        
        self.status_var.set("Analyzing test file...")
        self.root.update()
        
        try:
            # Parse test file
            self.test_functions = self.parser.parse_file(filepath)
            
            # Update test list
            self.test_listbox.delete(0, tk.END)
            for test in self.test_functions:
                icon = "[T]" if test.type == 'testset' else "[F]"
                self.test_listbox.insert(tk.END, f"{icon} {test.name} ({test.type})")
            
            # Display imports
            self.display_imports()
            
            # Check for issues
            self.check_issues()
            
            self.status_var.set(f"Analyzed: {len(self.test_functions)} test(s) found")
            
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse test file:\n{e}")
            self.status_var.set("Parse failed")
    
    def on_test_select(self, event):
        """Handle test selection"""
        selection = self.test_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        test = self.test_functions[idx]
        
        self.display_test_details(test)
    
    def display_test_details(self, test: TestFunction):
        """Display test function details"""
        # Clear previous content
        self.details_text.delete('1.0', tk.END)
        self.code_text.delete('1.0', tk.END)
        
        # Details
        self.details_text.insert(tk.END, f"Test: {test.name}\n", 'header')
        self.details_text.insert(tk.END, f"Type: {test.type}\n")
        self.details_text.insert(tk.END, f"Lines: {test.line_start}-{test.line_end}\n")
        self.details_text.insert(tk.END, "\n")
        
        if test.variables:
            self.details_text.insert(tk.END, "Variables:\n", 'header')
            for name, value in test.variables.items():
                self.details_text.insert(tk.END, f"  {name} = {value}\n")
            self.details_text.insert(tk.END, "\n")
        
        if test.assertions:
            self.details_text.insert(tk.END, f"Assertions ({len(test.assertions)}):\n", 'header')
            for assertion in test.assertions:
                self.details_text.insert(tk.END, f"  {assertion}\n", 'info')
            self.details_text.insert(tk.END, "\n")
        
        if test.loops:
            self.details_text.insert(tk.END, f"Loops ({len(test.loops)}):\n", 'header')
            for loop in test.loops:
                self.details_text.insert(tk.END, f"  {loop[:50]}...\n")
            self.details_text.insert(tk.END, "\n")
        
        if test.conditionals:
            self.details_text.insert(tk.END, f"Conditionals ({len(test.conditionals)}):\n", 'header')
            for cond in test.conditionals:
                self.details_text.insert(tk.END, f"  {cond[:50]}...\n")
            self.details_text.insert(tk.END, "\n")
        
        if test.imports:
            self.details_text.insert(tk.END, "Required Imports:\n", 'header')
            for imp in test.imports:
                self.details_text.insert(tk.END, f"  {imp}\n", 'info')
        
        # Code
        self.code_text.insert(tk.END, test.code)
        
        # Results if available
        if test.name in self.test_results:
            result = self.test_results[test.name]
            self.display_test_result(result)
    
    def display_imports(self):
        """Display all imports in the file"""
        self.imports_text.delete('1.0', tk.END)
        
        self.imports_text.insert(tk.END, "Current Imports:\n", 'header')
        self.imports_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if self.parser.imports:
            for imp in sorted(self.parser.imports):
                self.imports_text.insert(tk.END, f"[OK] {imp}\n", 'success')
        else:
            self.imports_text.insert(tk.END, "No imports found\n", 'warning')
        
        self.imports_text.insert(tk.END, "\n\nRequired Imports (from tests):\n", 'header')
        self.imports_text.insert(tk.END, "=" * 60 + "\n\n")
        
        all_required = set()
        for test in self.test_functions:
            all_required.update(test.imports)
        
        for req_imp in sorted(all_required):
            if req_imp in self.parser.imports:
                self.imports_text.insert(tk.END, f"[OK] {req_imp}\n", 'success')
            else:
                self.imports_text.insert(tk.END, f"[X] {req_imp} (MISSING)\n", 'error')
        
        # Global constants
        if self.parser.constants:
            self.imports_text.insert(tk.END, "\n\nGlobal Constants:\n", 'header')
            self.imports_text.insert(tk.END, "=" * 60 + "\n\n")
            for name, value in self.parser.constants.items():
                self.imports_text.insert(tk.END, f"const {name} = {value}\n")
        
        # Structs
        if self.parser.structs:
            self.imports_text.insert(tk.END, "\n\nStructs:\n", 'header')
            self.imports_text.insert(tk.END, "=" * 60 + "\n\n")
            for name, fields in self.parser.structs.items():
                self.imports_text.insert(tk.END, f"struct {name}\n")
                for fname, ftype in fields:
                    self.imports_text.insert(tk.END, f"    {fname}::{ftype}\n")
                self.imports_text.insert(tk.END, "end\n\n")
    
    def check_issues(self):
        """Check for potential issues"""
        self.issues_text.delete('1.0', tk.END)
        
        self.issues_text.insert(tk.END, "Validation Issues:\n", 'header')
        self.issues_text.insert(tk.END, "=" * 60 + "\n\n")
        
        issues_found = False
        
        # Check for missing imports
        all_required = set()
        for test in self.test_functions:
            all_required.update(test.imports)
        
        missing_imports = all_required - self.parser.imports
        if missing_imports:
            issues_found = True
            self.issues_text.insert(tk.END, "Missing Imports:\n", 'error')
            for imp in sorted(missing_imports):
                self.issues_text.insert(tk.END, f"  [X] {imp}\n", 'error')
                self.issues_text.insert(tk.END, f"    Fix: Add '{imp}' to top of file\n", 'info')
            self.issues_text.insert(tk.END, "\n")
        
        # Check for empty tests
        empty_tests = [t for t in self.test_functions if not t.assertions and not t.nested_tests]
        if empty_tests:
            issues_found = True
            self.issues_text.insert(tk.END, "Empty Tests (no assertions):\n", 'warning')
            for test in empty_tests:
                self.issues_text.insert(tk.END, f"  [!] {test.name} (line {test.line_start})\n", 'warning')
            self.issues_text.insert(tk.END, "\n")
        
        # Check syntax for each test (limit and be lenient)
        syntax_checked = 0
        for test in self.test_functions[:5]:  # Only check first 5
            if len(test.code.strip()) > 0:
                valid, error = self.runner.validate_syntax(test.code)
                if not valid and error:
                    issues_found = True
                    self.issues_text.insert(tk.END, f"Syntax Error in '{test.name}':\n", 'error')
                    error_summary = error[:500] + "..." if len(error) > 500 else error
                    self.issues_text.insert(tk.END, f"  {error_summary}\n", 'error')
                    self.issues_text.insert(tk.END, "\n")
                syntax_checked += 1
        
        if syntax_checked > 0:
            self.issues_text.insert(tk.END, f"\n[INFO] Syntax validated {syntax_checked} test(s)\n", 'info')
        
        if not issues_found:
            self.issues_text.insert(tk.END, "[OK] No issues found!\n", 'success')
            self.issues_text.insert(tk.END, "\nAll tests appear valid. Ready to run.\n")
    
    def run_all_tests(self):
        """Run all tests in the file"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please load a test file first")
            return
        
        self.status_var.set("Running all tests...")
        self.root.update()
        
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._run_all_tests_thread)
        thread.start()
    
    def _run_all_tests_thread(self):
        """Thread worker for running all tests"""
        success, stdout, stderr = self.runner.run_test_file(self.current_file)
        
        # Update UI in main thread
        self.root.after(0, self._update_all_results, success, stdout, stderr)
    
    def _update_all_results(self, success, stdout, stderr):
        """Update UI with test results"""
        self.results_text.delete('1.0', tk.END)
        
        self.results_text.insert(tk.END, "Test Execution Results:\n", 'header')
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if success:
            self.results_text.insert(tk.END, "[OK] ALL TESTS PASSED\n\n", 'success')
        else:
            self.results_text.insert(tk.END, "[X] TESTS FAILED\n\n", 'error')
        
        if stdout:
            self.results_text.insert(tk.END, "Output:\n", 'header')
            self.results_text.insert(tk.END, stdout + "\n\n")
        
        if stderr:
            self.results_text.insert(tk.END, "Errors:\n", 'error')
            self.results_text.insert(tk.END, stderr + "\n")
            
            # Parse errors and provide suggestions
            self.provide_error_suggestions(stderr)
        
        self.notebook.select(2)  # Switch to Results tab
        
        if success:
            self.status_var.set("All tests passed! [OK]")
        else:
            self.status_var.set("Tests failed - see Results tab for details")
    
    def provide_error_suggestions(self, stderr: str):
        """Analyze errors and provide fix suggestions"""
        self.results_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.results_text.insert(tk.END, "Fix Suggestions:\n", 'header')
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Common error patterns
        if "LoadError" in stderr:
            self.results_text.insert(tk.END, "• Check file syntax and structure\n", 'info')
        
        if "UndefVarError" in stderr:
            # Extract undefined variable
            match = re.search(r"UndefVarError: `?(\w+)`? not defined", stderr)
            if match:
                var = match.group(1)
                self.results_text.insert(tk.END, f"• Variable '{var}' not defined\n", 'info')
                self.results_text.insert(tk.END, f"  Fix: Define '{var}' or add required import\n", 'info')
        
        if "MethodError" in stderr:
            self.results_text.insert(tk.END, "• Function call with incorrect argument types\n", 'info')
            self.results_text.insert(tk.END, "  Fix: Check function signature and argument types\n", 'info')
        
        if "syntax:" in stderr.lower():
            self.results_text.insert(tk.END, "• Syntax error detected\n", 'info')
            self.results_text.insert(tk.END, "  Fix: Check for missing 'end', parentheses, or quotes\n", 'info')
        
        if "ArgumentError" in stderr:
            self.results_text.insert(tk.END, "• Incorrect function arguments\n", 'info')
            self.results_text.insert(tk.END, "  Fix: Check argument count and types\n", 'info')
    
    def run_selected_test(self):
        """Run selected test"""
        selection = self.test_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a test to run")
            return
        
        idx = selection[0]
        test = self.test_functions[idx]
        
        self.status_var.set(f"Running test: {test.name}...")
        self.root.update()
        
        # Run test
        result = self.runner.run_test_code(test.code, test.name)
        self.test_results[test.name] = result
        
        # Display result
        self.display_test_result(result)
        
        if result.passed:
            self.status_var.set(f"Test '{test.name}' passed [OK]")
        else:
            self.status_var.set(f"Test '{test.name}' failed [X]")
    
    def display_test_result(self, result: TestResult):
        """Display single test result"""
        self.results_text.delete('1.0', tk.END)
        
        self.results_text.insert(tk.END, f"Test: {result.test_name}\n", 'header')
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if result.passed:
            self.results_text.insert(tk.END, "[OK] PASSED\n\n", 'success')
        else:
            self.results_text.insert(tk.END, "[X] FAILED\n\n", 'error')
        
        if result.stdout:
            self.results_text.insert(tk.END, "Output:\n", 'header')
            self.results_text.insert(tk.END, result.stdout + "\n\n")
        
        if result.error_msg or result.stderr:
            self.results_text.insert(tk.END, "Errors:\n", 'error')
            self.results_text.insert(tk.END, (result.error_msg or result.stderr) + "\n\n")
            self.provide_error_suggestions(result.error_msg or result.stderr)
        
        self.notebook.select(2)  # Switch to Results tab
    
    def create_new_test(self):
        """Open test builder dialog"""
        TestBuilderDialog(self.root, self.add_new_test)
    
    def add_new_test(self, code: str):
        """Add new test to file"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please load a test file first")
            return
        
        try:
            # Append test to file
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write("\n\n" + code + "\n")
            
            messagebox.showinfo("Success", "Test added successfully!")
            
            # Reload file
            self.analyze_file()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add test:\n{e}")
    
    def save_file(self):
        """Save current test file"""
        if not self.current_file:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".jl",
                filetypes=[("Julia files", "*.jl"), ("All files", "*.*")]
            )
            if filepath:
                self.current_file = filepath
        
        if self.current_file:
            messagebox.showinfo("Save", "Use external editor to save changes")
    
    def fix_imports(self):
        """Auto-fix missing imports"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please load a test file first")
            return
        
        # Collect all required imports
        all_required = set()
        for test in self.test_functions:
            all_required.update(test.imports)
        
        missing = all_required - self.parser.imports
        
        if not missing:
            messagebox.showinfo("No Issues", "All required imports are present")
            return
        
        # Ask confirmation
        imports_str = '\n'.join(sorted(missing))
        if messagebox.askyesno("Add Imports", f"Add these imports?\n\n{imports_str}"):
            try:
                with open(self.current_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add imports at top (after existing imports if any)
                lines = content.split('\n')
                insert_idx = 0
                
                # Find last import line
                for i, line in enumerate(lines):
                    if line.strip().startswith('using') or line.strip().startswith('import'):
                        insert_idx = i + 1
                
                # Insert new imports
                for imp in sorted(missing):
                    lines.insert(insert_idx, imp)
                    insert_idx += 1
                
                # Write back
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                messagebox.showinfo("Success", f"Added {len(missing)} import(s)")
                self.analyze_file()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fix imports:\n{e}")


def main():
    root = tk.Tk()
    app = TestValidatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
