#!/usr/bin/env python3
"""
Repository Health Check Script

This script analyzes the POFJSP repository structure and identifies:
- Missing documentation
- Inconsistent naming conventions
- Large files that should be gitignored
- Unused or duplicate files
- Configuration issues

Usage:
    python scripts/maintenance/repo_health_check.py [--fix] [--verbose]
"""

import os
import sys
import argparse
from pathlib import Path
import re
from collections import defaultdict


class RepoHealthChecker:
    def __init__(self, repo_path='.', verbose=False):
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        self.issues = defaultdict(list)
        self.suggestions = defaultdict(list)
        
    def check_file_naming(self):
        """Check for consistent naming conventions."""
        print("[NAMING] Checking file naming conventions...")
        
        # Python files should use snake_case
        for py_file in self.repo_path.rglob('*.py'):
            if py_file.stem != py_file.stem.lower():
                self.issues['naming'].append(f"Python file not lowercase: {py_file}")
            
            if ' ' in py_file.stem:
                self.issues['naming'].append(f"Python file contains spaces: {py_file}")
        
        # Check for inconsistent directory naming
        for directory in self.repo_path.rglob('*'):
            if directory.is_dir() and ' ' in directory.name:
                self.issues['naming'].append(f"Directory contains spaces: {directory}")
    
    def check_documentation(self):
        """Check for missing or incomplete documentation."""
        print("[DOCS] Checking documentation completeness...")
        
        # Check for README files in major directories
        major_dirs = ['src', 'scripts', 'tests', 'examples']
        for dir_name in major_dirs:
            dir_path = self.repo_path / dir_name
            if dir_path.exists():
                readme_path = dir_path / 'README.md'
                if not readme_path.exists():
                    self.suggestions['docs'].append(f"Add README.md to {dir_name}/")
        
        # Check for docstrings in Python modules
        for py_file in (self.repo_path / 'src').rglob('*.py'):
            if py_file.stat().st_size > 1000:  # Only check substantial files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                            self.suggestions['docs'].append(f"Add module docstring to {py_file}")
                except UnicodeDecodeError:
                    pass
    
    def check_large_files(self):
        """Identify large files that might need attention."""
        print("[FILES] Checking for large files...")
        
        large_threshold = 10 * 1024 * 1024  # 10MB
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    if size > large_threshold:
                        self.issues['large_files'].append(f"Large file ({size//1024//1024}MB): {file_path}")
                except (OSError, PermissionError):
                    pass
    
    def check_duplicate_functionality(self):
        """Look for potentially duplicate scripts or functions."""
        print("[DUPLICATES] Checking for potential duplicates...")
        
        # Check for similar script names
        script_names = defaultdict(list)
        for script in (self.repo_path / 'scripts').rglob('*.py'):
            name_parts = script.stem.split('_')
            for part in name_parts:
                if len(part) > 3:  # Ignore short words
                    script_names[part].append(script)
        
        for name_part, scripts in script_names.items():
            if len(scripts) > 2:
                self.suggestions['duplicates'].append(
                    f"Multiple scripts with '{name_part}': {[s.name for s in scripts]}")
    
    def check_configuration_consistency(self):
        """Check configuration files for consistency."""
        print("[CONFIG] Checking configuration consistency...")
        
        config_dir = self.repo_path / 'conf'
        if config_dir.exists():
            # Check for YAML syntax (basic check)
            for yaml_file in config_dir.rglob('*.yaml'):
                try:
                    with open(yaml_file, 'r') as f:
                        content = f.read()
                        # Basic YAML validation
                        if content.count(':') < 1:
                            self.issues['config'].append(f"Possibly empty config: {yaml_file}")
                except Exception as e:
                    self.issues['config'].append(f"Cannot read config: {yaml_file} - {e}")
    
    def check_output_organization(self):
        """Check outputs directory organization."""
        print("[OUTPUTS] Checking outputs organization...")
        
        outputs_dir = self.repo_path / 'outputs'
        if outputs_dir.exists():
            subdirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
            
            # Check for date directories at root level
            date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
            date_dirs = [d for d in subdirs if date_pattern.match(d.name)]
            
            if date_dirs:
                self.suggestions['organization'].append(
                    f"Found {len(date_dirs)} date directories in outputs/. "
                    "Consider moving to outputs/reports/daily/")
            
            # Check for algorithm-specific result directories
            algo_patterns = ['ga_', 'sa_', 'tabu_', 'iaoa_', 'rl_']
            algo_dirs = []
            for d in subdirs:
                if any(d.name.startswith(pattern) for pattern in algo_patterns):
                    algo_dirs.append(d)
            
            if algo_dirs:
                self.suggestions['organization'].append(
                    f"Found algorithm-specific directories: {[d.name for d in algo_dirs]}. "
                    "Consider consolidating in outputs/algorithms/")
    
    def check_test_coverage(self):
        """Check for test files and coverage."""
        print("[TESTS] Checking test coverage...")
        
        src_files = list((self.repo_path / 'src').rglob('*.py'))
        test_files = list((self.repo_path / 'tests').rglob('test_*.py'))
        
        src_modules = set()
        for src_file in src_files:
            if src_file.name != '__init__.py':
                module_name = src_file.stem
                src_modules.add(module_name)
        
        test_modules = set()
        for test_file in test_files:
            if test_file.name.startswith('test_'):
                module_name = test_file.stem[5:]  # Remove 'test_' prefix
                test_modules.add(module_name)
        
        untested_modules = src_modules - test_modules
        if untested_modules:
            self.suggestions['testing'].append(
                f"Modules without tests: {sorted(untested_modules)}")
    
    def generate_gitignore_suggestions(self):
        """Suggest .gitignore additions."""
        print("[GITIGNORE] Checking .gitignore completeness...")
        
        gitignore_path = self.repo_path / '.gitignore'
        gitignore_content = ""
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
        
        # Check for common patterns that should be ignored
        suggestions = []
        
        # Python cache files
        if '__pycache__' not in gitignore_content:
            if any(self.repo_path.rglob('__pycache__')):
                suggestions.append("__pycache__/")
        
        # Large output files
        if '*.png' not in gitignore_content:
            png_files = list(self.repo_path.rglob('*.png'))
            if len(png_files) > 10:
                suggestions.append("# Large generated files\n*.png")
        
        # Model files
        if '*.pkl' not in gitignore_content:
            if any(self.repo_path.rglob('*.pkl')):
                suggestions.append("*.pkl")
        
        if suggestions:
            self.suggestions['gitignore'] = suggestions
    
    def run_all_checks(self):
        """Run all health checks."""
        print("[HEALTH] Running POFJSP Repository Health Check")
        print("=" * 50)
        
        self.check_file_naming()
        self.check_documentation()
        self.check_large_files()
        self.check_duplicate_functionality()
        self.check_configuration_consistency()
        self.check_output_organization()
        self.check_test_coverage()
        self.generate_gitignore_suggestions()
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate a health report."""
        print("\n" + "=" * 50)
        print("[REPORT] HEALTH CHECK REPORT")
        print("=" * 50)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        total_suggestions = sum(len(suggestions) for suggestions in self.suggestions.values())
        
        if total_issues == 0 and total_suggestions == 0:
            print("[SUCCESS] Repository is in excellent health!")
            return True
        
        # Report issues
        if total_issues > 0:
            print(f"\n[ISSUES] FOUND ({total_issues}):")
            for category, issue_list in self.issues.items():
                if issue_list:
                    print(f"\n  {category.upper()}:")
                    for issue in issue_list:
                        print(f"    - {issue}")
        
        # Report suggestions
        if total_suggestions > 0:
            print(f"\n[SUGGESTIONS] ({total_suggestions}):")
            for category, suggestion_list in self.suggestions.items():
                if suggestion_list:
                    print(f"\n  {category.upper()}:")
                    for suggestion in suggestion_list:
                        print(f"    - {suggestion}")
        
        # Summary
        health_score = max(0, 100 - (total_issues * 10) - (total_suggestions * 2))
        print(f"\n[SCORE] REPOSITORY HEALTH SCORE: {health_score}/100")
        
        if health_score >= 90:
            print("[EXCELLENT] Great job!")
        elif health_score >= 70:
            print("[GOOD] Minor improvements recommended")
        elif health_score >= 50:
            print("[FAIR] Several improvements needed")
        else:
            print("[ATTENTION] Multiple issues found")
        
        return health_score >= 70


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check repository health")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--repo-path', type=str, default='.',
                       help='Path to repository (default: current directory)')
    
    args = parser.parse_args()
    
    checker = RepoHealthChecker(args.repo_path, args.verbose)
    success = checker.run_all_checks()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 