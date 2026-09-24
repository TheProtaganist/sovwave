#!/usr/bin/env python3
"""
Julia Export/Import Validator GUI
==================================
Validates Julia module exports and imports between files.
Checks if exported functions/types are correctly imported and used.
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple


@dataclass
class ExportItem:
    """Represents an exported function, type, or constant"""
    name: str
    type: str  # 'function', 'struct', 'const', 'module'
    signature: str = ""
    line_number: int = 0
    params: List[Tuple[str, str]] = None  # [(param_name, param_type), ...]
    return_type: str = ""
    struct_fields: List[Tuple[str, str]] = None  # [(field_name, field_type), ...]
    
    def __post_init__(self):
        if self.params is None:
            self.params = []
        if self.struct_fields is None:
            self.struct_fields = []


@dataclass
class ImportItem:
    """Represents an imported item"""
    name: str
    module: str
    line_number: int = 0


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: str  # 'error', 'warning', 'info'
    file: str
    line: int
    message: str
    suggestion: str = ""  # Optional fix suggestion


class JuliaValidator:
    """Core validation logic for Julia exports and imports"""
    
    def __init__(self):
        self.exports: Dict[str, List[ExportItem]] = {}
        self.imports: Dict[str, List[ImportItem]] = {}
        self.usages: Dict[str, List[Tuple[str, int]]] = {}
    
    def extract_function_params(self, signature: str) -> Tuple[List[Tuple[str, str]], str]:
        """Extract parameter names and types from function signature"""
        params = []
        return_type = "Any"
        
        # Extract parameters from signature
        param_match = re.search(r'\(([^)]*)\)', signature)
        if param_match:
            param_str = param_match.group(1)
            if param_str.strip():
                # Split by comma, handling nested types
                param_parts = []
                depth = 0
                current = ""
                for char in param_str + ',':
                    if char in '{[(':
                        depth += 1
                        current += char
                    elif char in '}])':
                        depth -= 1
                        current += char
                    elif char == ',' and depth == 0:
                        if current.strip():
                            param_parts.append(current.strip())
                        current = ""
                    else:
                        current += char
                
                for param in param_parts:
                    # Match: name::Type or name
                    type_match = re.search(r'(\w+!?)\s*::\s*([^,]+)', param)
                    if type_match:
                        params.append((type_match.group(1), type_match.group(2).strip()))
                    else:
                        # Just parameter name, no type
                        name_match = re.search(r'(\w+!?)', param)
                        if name_match:
                            params.append((name_match.group(1), "Any"))
        
        # Extract return type
        return_match = re.search(r'\)::\s*(\w+(?:\{[^}]+\})?)', signature)
        if return_match:
            return_type = return_match.group(1)
        
        return params, return_type
    
    def extract_struct_fields(self, content: str, struct_name: str, start_line: int) -> List[Tuple[str, str]]:
        """Extract field names and types from struct definition"""
        fields = []
        lines = content.split('\n')
        in_struct = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i].strip()
            
            if f'struct {struct_name}' in line or f'mutable struct {struct_name}' in line:
                in_struct = True
                continue
            
            if in_struct:
                if line.startswith('end'):
                    break
                
                # Match: field::Type
                field_match = re.search(r'(\w+)\s*::\s*([^\s#]+)', line)
                if field_match:
                    fields.append((field_match.group(1), field_match.group(2)))
        
        return fields
        
    def parse_exports(self, file_path: str, content: str) -> List[ExportItem]:
        """Extract all exported items from a Julia module"""
        exports = []
        lines = content.split('\n')
        
        # Find export statements
        for i, line in enumerate(lines, 1):
            # Match: export FunctionName, TypeName, ...
            export_match = re.search(r'^\s*export\s+(.+?)(?:#|$)', line)  # Stop at comment or end
            if export_match:
                items = export_match.group(1).split(',')
                for item in items:
                    name = item.strip()
                    if name:  # Skip empty strings
                        exports.append(ExportItem(
                            name=name,
                            type='exported',
                            line_number=i
                        ))
        
        # Find function definitions (with or without !)
        func_pattern = r'^\s*function\s+(\w+!?)[\s\(]'
        for i, line in enumerate(lines, 1):
            func_match = re.search(func_pattern, line)
            if func_match and not line.strip().startswith('#'):
                func_name = func_match.group(1)
                
                # Check if this function is exported
                for exp in exports:
                    if exp.name == func_name:
                        exp.type = 'function'
                        exp.signature = line.strip()
                        # Extract parameters and return type
                        exp.params, exp.return_type = self.extract_function_params(line.strip())
                        break
        
        # Find short-form functions: name(...) = ...
        short_func_pattern = r'^\s*(\w+!?)\s*\([^)]*\)\s*='
        for i, line in enumerate(lines, 1):
            if not line.strip().startswith('#') and '=' in line:
                short_match = re.search(short_func_pattern, line)
                if short_match:
                    func_name = short_match.group(1)
                    # Check if this function is exported
                    for exp in exports:
                        if exp.name == func_name:
                            exp.type = 'function'
                            exp.signature = line.strip()
                            # Extract parameters
                            exp.params, exp.return_type = self.extract_function_params(line.strip())
                            break
        
        # Find struct definitions
        struct_pattern = r'^\s*(?:mutable\s+)?struct\s+(\w+)'
        for i, line in enumerate(lines, 1):
            struct_match = re.search(struct_pattern, line)
            if struct_match:
                struct_name = struct_match.group(1)
                for exp in exports:
                    if exp.name == struct_name:
                        exp.type = 'struct'
                        exp.signature = line.strip()
                        # Extract struct fields
                        exp.struct_fields = self.extract_struct_fields(content, struct_name, i)
                        break
        
        # Find const definitions
        const_pattern = r'^\s*const\s+(\w+)\s*='
        for i, line in enumerate(lines, 1):
            const_match = re.search(const_pattern, line)
            if const_match:
                const_name = const_match.group(1)
                for exp in exports:
                    if exp.name == const_name:
                        exp.type = 'const'
                        exp.signature = line.strip()
                        break
        
        return exports
    
    def parse_imports(self, file_path: str, content: str) -> List[ImportItem]:
        """Extract all import/using statements"""
        imports = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Match: using Module: item1, item2 (SELECTIVE - check this FIRST!)
            using_selective = re.search(r'^\s*using\s+\.?(\w+(?:\.\w+)*):\s*(.+?)(?:#|$)', line)
            if using_selective:
                module_name = using_selective.group(1)
                items = using_selective.group(2).split(',')
                for item in items:
                    item_name = item.strip()
                    if item_name:  # Skip empty strings
                        imports.append(ImportItem(
                            name=item_name,
                            module=module_name,
                            line_number=i
                        ))
            # Match: using .Module or using Module (WILDCARD - only if no colon)
            elif re.search(r'^\s*using\s+\.?(\w+(?:\.\w+)*)(?:\s|$)', line) and ':' not in line:
                using_match = re.search(r'^\s*using\s+\.?(\w+(?:\.\w+)*)', line)
                if using_match:
                    module_name = using_match.group(1)
                    imports.append(ImportItem(
                        name='*',  # using imports everything
                        module=module_name,
                        line_number=i
                    ))
            
            # Match: import Module
            import_match = re.search(r'^\s*import\s+\.?(\w+(?:\.\w+)*)', line)
            if import_match and ':' not in line:
                module_name = import_match.group(1)
                imports.append(ImportItem(
                    name='*',
                    module=module_name,
                    line_number=i
                ))
        
        return imports
    
    def find_usages(self, content: str, exported_names: Set[str]) -> Dict[str, List[int]]:
        """Find where exported items are used in the file"""
        usages = {name: [] for name in exported_names}
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith('#'):
                continue
            
            for name in exported_names:
                # Escape ! in regex pattern
                escaped_name = re.escape(name)
                # Match function calls: name(...)
                if re.search(rf'\b{escaped_name}\s*\(', line):
                    usages[name].append(i)
                # Match type usage: ::TypeName or name::TypeName
                elif re.search(rf'::\s*{escaped_name}\b', line):
                    usages[name].append(i)
                # Match general usage
                elif re.search(rf'\b{escaped_name}\b', line):
                    usages[name].append(i)
        
        return usages
    
    def validate(self, export_file: str, import_file: str) -> List[ValidationIssue]:
        """Validate imports against exports"""
        issues = []
        
        # Read files
        try:
            with open(export_file, 'r', encoding='utf-8') as f:
                export_content = f.read()
            with open(import_file, 'r', encoding='utf-8') as f:
                import_content = f.read()
        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                file='system',
                line=0,
                message=f"Failed to read files: {e}"
            ))
            return issues
        
        # Parse exports and imports
        exports = self.parse_exports(export_file, export_content)
        imports = self.parse_imports(import_file, import_content)
        
        export_names = {exp.name for exp in exports}
        
        # Find usages in import file
        usages = self.find_usages(import_content, export_names)
        
        # Also find ALL function calls in import file (not just exported ones)
        all_function_calls = set()
        for line in import_content.split('\n'):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            # Match function calls: name(...) or name!(...)
            func_calls = re.findall(r'\b(\w+!?)\s*\(', line)
            all_function_calls.update(func_calls)
        
        # Check 1: Are all used items properly imported?
        imported_names = set()
        importing_modules = set()
        
        for imp in imports:
            if imp.name == '*':
                importing_modules.add(imp.module)
            else:
                imported_names.add(imp.name)
        
        for name, usage_lines in usages.items():
            if usage_lines:  # Item is used
                if name not in imported_names and not importing_modules:
                    issues.append(ValidationIssue(
                        severity='error',
                        file=Path(import_file).name,
                        line=usage_lines[0],
                        message=f"'{name}' is used but not imported. Add 'using ModuleName: {name}' or 'using ModuleName'"
                    ))
        
        # Check 1b: Check for function calls that might be from the module but not exported
        # First, find functions defined in the import file itself
        local_functions = set()
        for line in import_content.split('\n'):
            # Find local function definitions
            if re.search(r'^\s*function\s+(\w+!?)[\s\(]', line):
                match = re.search(r'^\s*function\s+(\w+!?)[\s\(]', line)
                local_functions.add(match.group(1))
            elif re.search(r'^\s*(\w+!?)\s*\([^)]*\)\s*=', line):
                match = re.search(r'^\s*(\w+!?)\s*\([^)]*\)\s*=', line)
                if '::' not in line.split('(')[0]:
                    local_functions.add(match.group(1))
        
        for func_name in all_function_calls:
            if func_name not in imported_names and not importing_modules:
                # Skip Julia built-ins and local functions
                builtins = {
                    'println', 'print', 'error', 'warn', 'length', 'push!', 'pop!', 
                    'size', 'get', 'set', 'min', 'max', 'sum', 'prod', 'any', 'all',
                    'filter', 'map', 'reduce', 'collect', 'rand', 'randn', 'zeros',
                    'ones', 'fill', 'similar', 'copy', 'deepcopy', 'typeof', 'isa',
                    'convert', 'parse', 'string', 'join', 'split', 'strip', 'replace'
                }
                if func_name not in builtins and func_name not in export_names and func_name not in local_functions:
                    # Find line where it's used (skip comments)
                    for i, line in enumerate(import_content.split('\n'), 1):
                        if line.strip().startswith('#'):
                            continue
                        if re.search(rf'\b{re.escape(func_name)}\s*\(', line):
                            issues.append(ValidationIssue(
                                severity='error',
                                file=Path(import_file).name,
                                line=i,
                                message=f"'{func_name}' is used but not imported or not exported from module"
                            ))
                            break
        
        # Check 2: Are there unused imports?
        for imp in imports:
            if imp.name != '*':
                # Check if imported item is actually used in the code
                # Look for usage in import_content
                escaped_name = re.escape(imp.name)
                is_used = False
                for line in import_content.split('\n'):
                    if re.search(rf'\b{escaped_name}\s*\(', line) or \
                       re.search(rf'::\s*{escaped_name}\b', line) or \
                       (re.search(rf'\b{escaped_name}\b', line) and not line.strip().startswith('using')):
                        is_used = True
                        break
                
                if not is_used:
                    issues.append(ValidationIssue(
                        severity='warning',
                        file=Path(import_file).name,
                        line=imp.line_number,
                        message=f"'{imp.name}' is imported but never used"
                    ))
        
        # Check 3: Are exported items actually defined?
        defined_items = set()
        export_lines = export_content.split('\n')
        for i, line in enumerate(export_lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Find function definitions (both long and short form, with or without !)
            if re.search(r'^\s*function\s+(\w+!?)[\s\(]', line):
                match = re.search(r'^\s*function\s+(\w+!?)[\s\(]', line)
                defined_items.add(match.group(1))
            # Short function form: name(...) = ... or name!(...) = ...
            elif re.search(r'^\s*(\w+!?)\s*\([^)]*\)\s*=', line):
                match = re.search(r'^\s*(\w+!?)\s*\([^)]*\)\s*=', line)
                # Make sure it's not a variable assignment with ::Type
                if '::' not in line.split('(')[0]:
                    defined_items.add(match.group(1))
            # Find struct definitions
            if re.search(r'^\s*(?:mutable\s+)?struct\s+(\w+)', line):
                match = re.search(r'^\s*(?:mutable\s+)?struct\s+(\w+)', line)
                defined_items.add(match.group(1))
            # Find const definitions
            if re.search(r'^\s*const\s+(\w+)', line):
                match = re.search(r'^\s*const\s+(\w+)', line)
                defined_items.add(match.group(1))
        
        for exp in exports:
            if exp.name not in defined_items and exp.type == 'exported':
                issues.append(ValidationIssue(
                    severity='error',
                    file=Path(export_file).name,
                    line=exp.line_number,
                    message=f"'{exp.name}' is exported but not defined in the module",
                    suggestion=f"Define '{exp.name}' or remove from export list"
                ))
        
        # Check 4: Type consistency and parameter validation
        self.check_function_calls(import_content, exports, issues, Path(import_file).name)
        
        return issues
    
    def check_function_calls(self, import_content: str, exports: List[ExportItem], 
                            issues: List[ValidationIssue], filename: str):
        """Check if function calls match their definitions"""
        lines = import_content.split('\n')
        
        # Build function signature map
        func_map = {exp.name: exp for exp in exports if exp.type == 'function'}
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('#'):
                continue
            
            # Find function calls: name(args)
            for func_name, func_def in func_map.items():
                escaped_name = re.escape(func_name)
                call_pattern = rf'\b{escaped_name}\s*\(([^)]*)\)'
                call_match = re.search(call_pattern, line)
                
                if call_match:
                    args_str = call_match.group(1).strip()
                    
                    # Count arguments (simple heuristic)
                    if args_str:
                        # Split by comma, accounting for nested parens
                        arg_count = 1
                        depth = 0
                        for char in args_str:
                            if char in '([{':
                                depth += 1
                            elif char in ')]}':
                                depth -= 1
                            elif char == ',' and depth == 0:
                                arg_count += 1
                    else:
                        arg_count = 0
                    
                    expected_count = len(func_def.params)
                    
                    # Check argument count
                    if arg_count != expected_count:
                        param_list = ", ".join([f"{name}::{typ}" for name, typ in func_def.params])
                        issues.append(ValidationIssue(
                            severity='error',
                            file=filename,
                            line=i,
                            message=f"'{func_name}' expects {expected_count} argument(s) but got {arg_count}",
                            suggestion=f"Expected signature: {func_name}({param_list})"
                        ))
                    
                    # Check argument types (basic heuristic)
                    if arg_count == expected_count and func_def.params:
                        args_list = []
                        current_arg = ""
                        depth = 0
                        for char in args_str + ',':
                            if char in '([{':
                                depth += 1
                                current_arg += char
                            elif char in ')]}':
                                depth -= 1
                                current_arg += char
                            elif char == ',' and depth == 0:
                                if current_arg.strip():
                                    args_list.append(current_arg.strip())
                                current_arg = ""
                            else:
                                current_arg += char
                        
                        for j, (arg, (param_name, param_type)) in enumerate(zip(args_list, func_def.params)):
                            if param_type != "Any":
                                # Basic type inference
                                inferred_type = self.infer_type(arg)
                                if inferred_type and not self.types_compatible(inferred_type, param_type):
                                    issues.append(ValidationIssue(
                                        severity='warning',
                                        file=filename,
                                        line=i,
                                        message=f"'{func_name}' parameter {j+1} ('{param_name}') expects {param_type} but got {inferred_type}",
                                        suggestion=f"Convert argument to {param_type} or check type compatibility"
                                    ))
    
    def infer_type(self, arg: str) -> str:
        """Infer Julia type from argument value"""
        arg = arg.strip()
        
        # String literals
        if arg.startswith('"') and arg.endswith('"'):
            return "String"
        if arg.startswith("'") and arg.endswith("'"):
            return "Char"
        
        # Numeric literals
        if re.match(r'^-?\d+$', arg):
            return "Int64"
        if re.match(r'^-?\d+\.\d+$', arg):
            return "Float64"
        
        # Boolean
        if arg in ['true', 'false']:
            return "Bool"
        
        # Arrays
        if arg.startswith('[') and arg.endswith(']'):
            # Check if it's typed array
            if re.match(r'^\w+\[', arg):
                type_match = re.match(r'^(\w+)\[', arg)
                return f"Vector{{{type_match.group(1)}}}"
            else:
                # Try to infer from first element
                inner = arg[1:-1].strip()
                if inner:
                    first_elem = inner.split(',')[0].strip()
                    elem_type = self.infer_type(first_elem)
                    if elem_type:
                        return f"Vector{{{elem_type}}}"
            return "Vector"
        
        # Nothing/missing
        if arg == 'nothing':
            return "Nothing"
        
        # Unknown - could be variable
        return None
    
    def types_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type"""
        if actual == expected:
            return True
        
        # Any accepts everything
        if expected == "Any":
            return True
        
        # Abstract number types
        if expected in ["Number", "Real", "Integer"]:
            if actual in ["Int64", "Int32", "Float64", "Float32", "UInt64"]:
                return True
        
        if expected in ["AbstractFloat", "Real"]:
            if actual in ["Float64", "Float32"]:
                return True
        
        if expected in ["Integer", "Real"]:
            if actual in ["Int64", "Int32", "UInt64"]:
                return True
        
        # AbstractArray, AbstractVector
        if "AbstractArray" in expected or "AbstractVector" in expected:
            if "Vector" in actual or "Array" in actual:
                return True
        
        # Vector type matching
        if expected.startswith("Vector{") and actual.startswith("Vector{"):
            exp_inner = re.search(r'Vector\{([^}]+)\}', expected)
            act_inner = re.search(r'Vector\{([^}]+)\}', actual)
            if exp_inner and act_inner:
                return self.types_compatible(act_inner.group(1), exp_inner.group(1))
        
        return False
        
        # Check 4: Type consistency (basic check)
        for exp in exports:
            if exp.type == 'function' and exp.name in usages:
                for usage_line in usages[exp.name]:
                    # Could add more sophisticated type checking here
                    pass
        
        return issues


class ValidatorGUI:
    """Tkinter GUI for Julia Export/Import Validator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Julia Export/Import Validator")
        self.root.geometry("1000x700")
        
        self.validator = JuliaValidator()
        self.export_file = ""
        self.import_file = ""
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="Julia Export/Import Validator", 
                        )
        title.pack(pady=(0, 10))
        
        # Export file selection
        export_frame = tk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=5)
        tk.Label(export_frame, text="Export File:").pack(side=tk.LEFT)
        self.export_entry = tk.Entry(export_frame, width=50)
        self.export_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(export_frame, text="Browse", command=self.browse_export).pack(side=tk.LEFT)
        
        # Import file selection
        import_frame = tk.Frame(main_frame)
        import_frame.pack(fill=tk.X, pady=5)
        tk.Label(import_frame, text="Import File:").pack(side=tk.LEFT)
        self.import_entry = tk.Entry(import_frame, width=50)
        self.import_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(import_frame, text="Browse", command=self.browse_import).pack(side=tk.LEFT)
        
        # Validate button
        tk.Button(main_frame, text="Validate", command=self.validate,
                 bg='#4CAF50', fg='white', padx=20, pady=5).pack(pady=10)
        
        # Notebook for tabs
        self.notebook_frame = tk.Frame(main_frame)
        self.notebook_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab buttons
        tab_frame = tk.Frame(self.notebook_frame)
        tab_frame.pack(fill=tk.X)
        
        self.current_tab = 0
        self.tab_buttons = []
        tabs = ["Issues", "Exports", "Imports", "Checklist"]
        for i, tab_name in enumerate(tabs):
            btn = tk.Button(tab_frame, text=tab_name, command=lambda x=i: self.show_tab(x))
            btn.pack(side=tk.LEFT, padx=2)
            self.tab_buttons.append(btn)
        
        # Tab content frames
        self.tab_frames = []
        self.text_widgets = []
        
        for i in range(4):
            frame = tk.Frame(self.notebook_frame)
            text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Configure tags
            text_widget.tag_config('error', foreground='red')
            text_widget.tag_config('warning', foreground='orange')
            text_widget.tag_config('success', foreground='green')
            
            self.tab_frames.append(frame)
            self.text_widgets.append(text_widget)
        
        self.issues_text = self.text_widgets[0]
        self.exports_text = self.text_widgets[1]
        self.imports_text = self.text_widgets[2]
        self.checklist_text = self.text_widgets[3]
        
        self.show_tab(0)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W, bg='lightgray')
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def show_tab(self, index):
        """Show the selected tab"""
        for i, frame in enumerate(self.tab_frames):
            if i == index:
                frame.pack(fill=tk.BOTH, expand=True)
                self.tab_buttons[i].config(relief=tk.SUNKEN, bg='lightblue')
            else:
                frame.pack_forget()
                self.tab_buttons[i].config(relief=tk.RAISED, bg='lightgray')
        self.current_tab = index
    
    def browse_export(self):
        """Browse for export file"""
        filename = filedialog.askopenfilename(
            title="Select Export File (Module)",
            filetypes=[("Julia files", "*.jl"), ("All files", "*.*")]
        )
        if filename:
            self.export_file = filename
            self.export_entry.delete(0, tk.END)
            self.export_entry.insert(0, filename)
            self.status_var.set(f"Selected export file: {Path(filename).name}")
    
    def browse_import(self):
        """Browse for import file"""
        filename = filedialog.askopenfilename(
            title="Select Import File (Using)",
            filetypes=[("Julia files", "*.jl"), ("All files", "*.*")]
        )
        if filename:
            self.import_file = filename
            self.import_entry.delete(0, tk.END)
            self.import_entry.insert(0, filename)
            self.status_var.set(f"Selected import file: {Path(filename).name}")
    
    def validate(self):
        """Run validation"""
        self.export_file = self.export_entry.get()
        self.import_file = self.import_entry.get()
        
        if not self.export_file or not self.import_file:
            messagebox.showwarning("Missing Files", 
                                  "Please select both export and import files")
            return
        
        self.status_var.set("Validating...")
        self.root.update()
        
        # Clear previous results
        self.issues_text.delete(1.0, tk.END)
        self.exports_text.delete(1.0, tk.END)
        self.imports_text.delete(1.0, tk.END)
        self.checklist_text.delete(1.0, tk.END)
        
        try:
            # Read files
            with open(self.export_file, 'r', encoding='utf-8') as f:
                export_content = f.read()
            with open(self.import_file, 'r', encoding='utf-8') as f:
                import_content = f.read()
            
            # Parse
            exports = self.validator.parse_exports(self.export_file, export_content)
            imports = self.validator.parse_imports(self.import_file, import_content)
            
            # Display exports
            self.exports_text.insert(tk.END, f"Exports from {Path(self.export_file).name}\n")
            self.exports_text.insert(tk.END, "=" * 70 + "\n\n")
            for exp in exports:
                self.exports_text.insert(tk.END, f"[{exp.type.upper()}] {exp.name}\n")
                if exp.signature:
                    self.exports_text.insert(tk.END, f"  Line {exp.line_number}: {exp.signature}\n")
                if exp.params:
                    param_str = ", ".join([f"{name}::{typ}" for name, typ in exp.params])
                    self.exports_text.insert(tk.END, f"  Parameters: {param_str}\n")
                if exp.return_type:
                    self.exports_text.insert(tk.END, f"  Returns: {exp.return_type}\n")
                if exp.struct_fields:
                    self.exports_text.insert(tk.END, f"  Fields:\n")
                    for fname, ftype in exp.struct_fields:
                        self.exports_text.insert(tk.END, f"    {fname}::{ftype}\n")
                self.exports_text.insert(tk.END, "\n")
            
            # Display imports
            self.imports_text.insert(tk.END, f"Imports in {Path(self.import_file).name}\n")
            self.imports_text.insert(tk.END, "=" * 70 + "\n\n")
            for imp in imports:
                if imp.name == '*':
                    self.imports_text.insert(tk.END, f"using {imp.module} (all exports)\n")
                else:
                    self.imports_text.insert(tk.END, f"using {imp.module}: {imp.name}\n")
                self.imports_text.insert(tk.END, f"  Line {imp.line_number}\n\n")
            
            # Validate
            issues = self.validator.validate(self.export_file, self.import_file)
            
            # Display issues
            if not issues:
                self.issues_text.insert(tk.END, "[OK] No issues found!\n", 'success')
                self.issues_text.insert(tk.END, "\nAll exports and imports are correctly matched.\n")
            else:
                errors = [i for i in issues if i.severity == 'error']
                warnings = [i for i in issues if i.severity == 'warning']
                
                self.issues_text.insert(tk.END, f"Found {len(errors)} errors, {len(warnings)} warnings\n\n")
                
                if errors:
                    self.issues_text.insert(tk.END, "ERRORS:\n", 'error')
                    self.issues_text.insert(tk.END, "-" * 70 + "\n")
                    for issue in errors:
                        self.issues_text.insert(tk.END, f"[{issue.file}:{issue.line}] ", 'error')
                        self.issues_text.insert(tk.END, f"{issue.message}\n")
                        if issue.suggestion:
                            self.issues_text.insert(tk.END, f"  > Suggestion: {issue.suggestion}\n", 'info')
                        self.issues_text.insert(tk.END, "\n")
                
                if warnings:
                    self.issues_text.insert(tk.END, "\nWARNINGS:\n", 'warning')
                    self.issues_text.insert(tk.END, "-" * 70 + "\n")
                    for issue in warnings:
                        self.issues_text.insert(tk.END, f"[{issue.file}:{issue.line}] ", 'warning')
                        self.issues_text.insert(tk.END, f"{issue.message}\n")
                        if issue.suggestion:
                            self.issues_text.insert(tk.END, f"  > Suggestion: {issue.suggestion}\n", 'info')
                        self.issues_text.insert(tk.END, "\n")
            
            # Generate checklist
            self.generate_checklist(exports, imports, issues)
            
            self.status_var.set(f"Validation complete: {len(issues)} issue(s) found")
            
        except Exception as e:
            self.issues_text.insert(tk.END, f"ERROR: {e}\n", 'error')
            self.status_var.set("Validation failed")
            messagebox.showerror("Validation Error", str(e))
    
    def generate_checklist(self, exports, imports, issues):
        """Generate a checklist for both files"""
        export_file_name = Path(self.export_file).name
        import_file_name = Path(self.import_file).name
        
        # Checklist header
        self.checklist_text.insert(tk.END, "VALIDATION CHECKLIST\n")
        self.checklist_text.insert(tk.END, "=" * 70 + "\n\n")
        
        # Export file checklist
        self.checklist_text.insert(tk.END, f"Export File: {export_file_name}\n")
        self.checklist_text.insert(tk.END, "-" * 70 + "\n")
        
        export_errors = [i for i in issues if export_file_name in i.file and i.severity == 'error']
        
        if export_errors:
            for issue in export_errors:
                self.checklist_text.insert(tk.END, f"[ ] {issue.message}\n")
        else:
            self.checklist_text.insert(tk.END, "[OK] All exports are properly defined\n")
        
        self.checklist_text.insert(tk.END, f"[OK] Total exports: {len(exports)}\n")
        
        # Import file checklist
        self.checklist_text.insert(tk.END, f"\nImport File: {import_file_name}\n")
        self.checklist_text.insert(tk.END, "-" * 70 + "\n")
        
        import_issues = [i for i in issues if import_file_name in i.file]
        
        if import_issues:
            for issue in import_issues:
                marker = "[ ]" if issue.severity == 'error' else "[!]"
                self.checklist_text.insert(tk.END, f"{marker} {issue.message}\n")
        else:
            self.checklist_text.insert(tk.END, "[OK] All imports are correct\n")
        
        self.checklist_text.insert(tk.END, f"[OK] Total imports: {len(imports)}\n")
        
        # Summary
        self.checklist_text.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.checklist_text.insert(tk.END, "SUMMARY\n")
        self.checklist_text.insert(tk.END, "=" * 70 + "\n")
        
        total_errors = len([i for i in issues if i.severity == 'error'])
        total_warnings = len([i for i in issues if i.severity == 'warning'])
        
        if total_errors == 0 and total_warnings == 0:
            self.checklist_text.insert(tk.END, "[OK] All checks passed! Files are properly synchronized.\n")
        else:
            self.checklist_text.insert(tk.END, f"Found {total_errors} error(s) and {total_warnings} warning(s)\n")
            self.checklist_text.insert(tk.END, "\nAction Items:\n")
            if total_errors > 0:
                self.checklist_text.insert(tk.END, f"1. Fix {total_errors} critical error(s)\n")
            if total_warnings > 0:
                self.checklist_text.insert(tk.END, f"2. Review {total_warnings} warning(s)\n")


def main():
    root = tk.Tk()
    app = ValidatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
