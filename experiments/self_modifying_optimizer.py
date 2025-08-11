#!/usr/bin/env python3
"""
Self-Modifying Optimization Framework

This framework can:
1. Detect when parameters aren't the problem - the logic is
2. Modify the actual strategy code to fix logical issues
3. Update documentation to reflect changes
4. Test that modifications actually improve things
5. Keep a history of all changes for rollback if needed
"""

import os
import sys
import ast
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class CodeModification:
    """Record of a code modification"""
    timestamp: str
    file_path: str
    line_number: int
    old_code: str
    new_code: str
    reason: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    success: Optional[bool] = None
    rollback_id: Optional[str] = None


@dataclass
class LogicIssue:
    """Identified issue with strategy logic"""
    issue_type: str  # "hardcoded", "wrong_operator", "missing_condition", etc.
    location: str  # file:line
    current_logic: str
    problem: str
    proposed_fix: str
    confidence: float


class SelfModifyingOptimizer:
    """Framework that can modify its own logic based on learnings"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.modifications_dir = Path(__file__).parent / "code_modifications"
        self.modifications_dir.mkdir(exist_ok=True)
        
        # Backup directory for rollbacks
        self.backup_dir = self.modifications_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Modification history
        self.history_file = self.modifications_dir / "modification_history.json"
        self.issues_file = self.modifications_dir / "identified_issues.json"
        
        self.modification_history = self._load_history()
        self.identified_issues = self._load_issues()
        
        # Strategy file locations
        self.strategy_files = {
            'phase2_divergence': self.base_dir / 'strategies/squeezeflow/components/phase2_divergence.py',
            'phase3_reset': self.base_dir / 'strategies/squeezeflow/components/phase3_reset.py',
            'phase4_scoring': self.base_dir / 'strategies/squeezeflow/components/phase4_scoring.py',
            'config': self.base_dir / 'strategies/squeezeflow/config.py',
            'strategy': self.base_dir / 'strategies/squeezeflow/strategy.py'
        }
    
    def _load_history(self) -> List[CodeModification]:
        """Load modification history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return [CodeModification(**mod) for mod in data]
        return []
    
    def _load_issues(self) -> List[LogicIssue]:
        """Load identified issues"""
        if self.issues_file.exists():
            with open(self.issues_file, 'r') as f:
                data = json.load(f)
                return [LogicIssue(**issue) for issue in data]
        return []
    
    def analyze_code_for_issues(self, validation_results: Dict[str, Any]) -> List[LogicIssue]:
        """
        Analyze code to find logical issues based on validation results
        
        Example: If TON can't trade because of hardcoded threshold,
        this finds the exact line and proposes a fix.
        """
        
        issues = []
        
        # Check for hardcoded threshold issue
        if 'TON' in validation_results and validation_results['TON'].get('blocked_by_threshold'):
            # Read the divergence file
            with open(self.strategy_files['phase2_divergence'], 'r') as f:
                lines = f.readlines()
            
            # Find the hardcoded threshold
            for i, line in enumerate(lines, 1):
                if 'min_change_threshold = 1e6' in line:
                    issue = LogicIssue(
                        issue_type='hardcoded',
                        location=f'phase2_divergence.py:{i}',
                        current_logic='min_change_threshold = 1e6',
                        problem='Hardcoded 1M threshold blocks low-volume symbols',
                        proposed_fix='min_change_threshold = self._calculate_dynamic_threshold(symbol, recent_data)',
                        confidence=0.95
                    )
                    issues.append(issue)
                    break
        
        # Check for wrong comparison operators
        if validation_results.get('divergence_detection_inverted'):
            with open(self.strategy_files['phase2_divergence'], 'r') as f:
                content = f.read()
            
            # Parse AST to find comparisons
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare):
                    # Check if this is a spot/futures comparison
                    if hasattr(node, 'lineno'):
                        line = content.split('\n')[node.lineno - 1]
                        if 'spot' in line and 'futures' in line:
                            issue = LogicIssue(
                                issue_type='wrong_operator',
                                location=f'phase2_divergence.py:{node.lineno}',
                                current_logic=line.strip(),
                                problem='Comparison might be inverted',
                                proposed_fix='Check if spot and futures move in OPPOSITE directions',
                                confidence=0.7
                            )
                            issues.append(issue)
        
        # Check for missing conditions
        if validation_results.get('missing_volume_validation'):
            issue = LogicIssue(
                issue_type='missing_condition',
                location='phase2_divergence.py:_detect_divergence',
                current_logic='No volume validation',
                problem='Divergence detection doesn\'t validate volume significance',
                proposed_fix='Add volume validation before confirming divergence',
                confidence=0.8
            )
            issues.append(issue)
        
        return issues
    
    def create_backup(self, file_path: Path) -> str:
        """Create backup before modification"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_id = f"{file_path.stem}_{timestamp}"
        backup_path = self.backup_dir / f"{backup_id}.py"
        
        shutil.copy(file_path, backup_path)
        return backup_id
    
    def modify_code(self, issue: LogicIssue, dry_run: bool = False) -> CodeModification:
        """
        Actually modify the code to fix an issue
        
        Args:
            issue: The identified issue to fix
            dry_run: If True, show what would be changed without modifying
        """
        
        # Parse location
        file_name, line_num = issue.location.split(':')
        line_num = int(line_num) if line_num.isdigit() else -1
        
        # Get the actual file path
        file_key = file_name.replace('.py', '')
        if file_key not in self.strategy_files:
            raise ValueError(f"Unknown file: {file_name}")
        
        file_path = self.strategy_files[file_key]
        
        # Create backup
        backup_id = self.create_backup(file_path) if not dry_run else "dry_run"
        
        # Read current content
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find and modify the problematic line
        old_code = ""
        new_code = ""
        
        if issue.issue_type == 'hardcoded':
            # Replace hardcoded value with dynamic calculation
            for i, line in enumerate(lines):
                if issue.current_logic in line:
                    old_code = line
                    
                    # Generate dynamic replacement
                    if 'min_change_threshold = 1e6' in line:
                        # Add method to calculate dynamic threshold
                        new_code = self._generate_dynamic_threshold_code(line)
                        lines[i] = new_code
                        
                        # Also need to add the helper method
                        lines = self._add_helper_method(lines, file_key)
                    break
        
        elif issue.issue_type == 'wrong_operator':
            # Fix comparison operators
            if line_num > 0 and line_num <= len(lines):
                old_code = lines[line_num - 1]
                # Invert the comparison
                new_code = old_code.replace('>', '<') if '>' in old_code else old_code.replace('<', '>')
                lines[line_num - 1] = new_code
        
        elif issue.issue_type == 'missing_condition':
            # Add missing validation
            # This is more complex and would need careful insertion
            pass
        
        # Create modification record
        modification = CodeModification(
            timestamp=datetime.now().isoformat(),
            file_path=str(file_path),
            line_number=line_num,
            old_code=old_code.strip(),
            new_code=new_code.strip() if new_code else issue.proposed_fix,
            reason=issue.problem,
            expected_outcome=f"Fix: {issue.problem}",
            rollback_id=backup_id
        )
        
        if not dry_run:
            # Write modified content
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
            # Save modification history
            self.modification_history.append(modification)
            self._save_history()
            
            print(f"✅ Modified {file_name}:{line_num}")
            print(f"   Old: {old_code.strip()}")
            print(f"   New: {new_code.strip()}")
        else:
            print(f"DRY RUN - Would modify {file_name}:{line_num}")
            print(f"   Old: {old_code.strip()}")
            print(f"   New: {new_code.strip() if new_code else issue.proposed_fix}")
        
        return modification
    
    def _generate_dynamic_threshold_code(self, original_line: str) -> str:
        """Generate dynamic threshold calculation code"""
        
        indent = len(original_line) - len(original_line.lstrip())
        
        # Instead of hardcoded value, calculate based on data
        new_code = ' ' * indent + '# MODIFIED: Dynamic threshold based on symbol characteristics\n'
        new_code += ' ' * indent + 'min_change_threshold = self._get_dynamic_threshold(spot_cvd, futures_cvd)\n'
        
        return new_code
    
    def _add_helper_method(self, lines: List[str], file_key: str) -> List[str]:
        """Add helper method for dynamic calculations"""
        
        if file_key == 'phase2_divergence':
            # Find the class definition
            class_line = -1
            for i, line in enumerate(lines):
                if 'class Phase2Divergence' in line:
                    class_line = i
                    break
            
            if class_line >= 0:
                # Find where to insert (after __init__ method)
                insert_line = -1
                indent_level = 0
                
                for i in range(class_line, len(lines)):
                    if 'def __init__' in lines[i]:
                        # Find the end of __init__
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and not lines[j].startswith(' '):
                                break
                            if 'def ' in lines[j] and '__init__' not in lines[j]:
                                insert_line = j
                                indent_level = len(lines[j]) - len(lines[j].lstrip())
                                break
                
                if insert_line > 0:
                    # Insert the new method
                    method_code = self._generate_threshold_method(indent_level)
                    
                    # Insert before the next method
                    for line in reversed(method_code.split('\n')):
                        lines.insert(insert_line, line + '\n')
        
        return lines
    
    def _generate_threshold_method(self, indent: int) -> str:
        """Generate the dynamic threshold calculation method"""
        
        spaces = ' ' * indent
        code = f"""
{spaces}def _get_dynamic_threshold(self, spot_cvd: pd.Series, futures_cvd: pd.Series) -> float:
{spaces}    '''
{spaces}    Calculate dynamic threshold based on actual data characteristics
{spaces}    GENERATED BY SELF-MODIFYING OPTIMIZER
{spaces}    '''
{spaces}    import numpy as np
{spaces}    
{spaces}    # Calculate recent volatility
{spaces}    if len(spot_cvd) > 100:
{spaces}        recent_changes = spot_cvd.diff().abs().dropna()
{spaces}        
{spaces}        # Use 75th percentile as signal threshold
{spaces}        signal_threshold = np.percentile(recent_changes, 75)
{spaces}        
{spaces}        # Minimum threshold to avoid noise
{spaces}        min_threshold = np.percentile(recent_changes, 50) * 2
{spaces}        
{spaces}        return max(signal_threshold, min_threshold)
{spaces}    else:
{spaces}        # Fallback for insufficient data
{spaces}        return 1e5  # 100K default
"""
        return code.strip()
    
    def test_modification(self, modification: CodeModification) -> bool:
        """
        Test if a modification actually improves things
        
        Runs a quick backtest to see if the change helps
        """
        
        print(f"Testing modification: {modification.reason}")
        
        # Run a test backtest
        cmd = [
            "python3", "backtest/engine.py",
            "--symbol", "TON",  # Test on problematic symbol
            "--start-date", "2025-08-10",
            "--end-date", "2025-08-10",
            "--timeframe", "1s",
            "--balance", "10000",
            "--leverage", "1.0",
            "--strategy", "SqueezeFlowStrategy"
        ]
        
        env = os.environ.copy()
        env['INFLUX_HOST'] = '213.136.75.120'
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
                cwd=str(self.base_dir)
            )
            
            # Check if trades were generated (success metric)
            output = result.stdout + result.stderr
            
            if 'Total trades: 0' in output:
                print("❌ Modification didn't help - still no trades")
                modification.actual_outcome = "No improvement - 0 trades"
                modification.success = False
                return False
            elif 'Total trades:' in output:
                # Extract trade count
                import re
                match = re.search(r'Total trades:\s*(\d+)', output)
                if match:
                    trades = int(match.group(1))
                    print(f"✅ Modification worked - {trades} trades generated")
                    modification.actual_outcome = f"Success - {trades} trades"
                    modification.success = True
                    return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            modification.actual_outcome = f"Test error: {e}"
            modification.success = False
        
        return False
    
    def rollback_modification(self, modification: CodeModification):
        """Rollback a modification if it didn't work"""
        
        if modification.rollback_id and modification.rollback_id != "dry_run":
            backup_path = self.backup_dir / f"{modification.rollback_id}.py"
            
            if backup_path.exists():
                # Restore from backup
                shutil.copy(backup_path, modification.file_path)
                print(f"✅ Rolled back modification to {modification.file_path}")
                
                # Mark as rolled back
                modification.actual_outcome = "Rolled back"
                modification.success = False
                self._save_history()
    
    def update_documentation(self, modifications: List[CodeModification]):
        """
        Update documentation to reflect code changes
        """
        
        docs_file = self.base_dir / 'SYSTEM_TRUTH.md'
        
        # Read current docs
        with open(docs_file, 'r') as f:
            content = f.read()
        
        # Add modification notes
        mod_section = "\n\n## Recent Automated Modifications\n\n"
        mod_section += f"Last updated: {datetime.now().isoformat()}\n\n"
        
        for mod in modifications[-5:]:  # Last 5 modifications
            if mod.success:
                mod_section += f"### ✅ {mod.reason}\n"
                mod_section += f"- File: {Path(mod.file_path).name}\n"
                mod_section += f"- Change: {mod.old_code} → {mod.new_code}\n"
                mod_section += f"- Result: {mod.actual_outcome}\n\n"
        
        # Update or append section
        if "## Recent Automated Modifications" in content:
            # Replace existing section
            start = content.index("## Recent Automated Modifications")
            end = content.index("\n## ", start + 1) if "\n## " in content[start + 1:] else len(content)
            content = content[:start] + mod_section + content[end:]
        else:
            # Append new section
            content += mod_section
        
        # Write updated docs
        with open(docs_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated documentation with {len(modifications)} modifications")
    
    def _save_history(self):
        """Save modification history"""
        data = [asdict(mod) for mod in self.modification_history]
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_issues(self):
        """Save identified issues"""
        data = [asdict(issue) for issue in self.identified_issues]
        with open(self.issues_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run_adaptive_modification(self, validation_results: Dict[str, Any]):
        """
        Main entry point: Analyze, modify, test, and adapt
        """
        
        print("="*60)
        print("SELF-MODIFYING OPTIMIZATION")
        print("="*60)
        
        # 1. Analyze code for issues
        print("\n1. Analyzing code for logical issues...")
        issues = self.analyze_code_for_issues(validation_results)
        
        if not issues:
            print("   No issues found in code logic")
            return
        
        print(f"   Found {len(issues)} potential issues")
        
        for issue in issues:
            print(f"\n   Issue: {issue.problem}")
            print(f"   Location: {issue.location}")
            print(f"   Proposed fix: {issue.proposed_fix}")
            print(f"   Confidence: {issue.confidence:.0%}")
        
        # 2. Apply modifications (high confidence first)
        issues.sort(key=lambda x: x.confidence, reverse=True)
        
        for issue in issues:
            if issue.confidence > 0.8:  # Only apply high-confidence fixes
                print(f"\n2. Applying modification for: {issue.problem}")
                
                # Modify code
                modification = self.modify_code(issue, dry_run=False)
                
                # Test modification
                print("\n3. Testing modification...")
                success = self.test_modification(modification)
                
                if not success:
                    print("4. Modification didn't improve - rolling back")
                    self.rollback_modification(modification)
                else:
                    print("4. Modification successful - keeping changes")
                    
                    # Update documentation
                    self.update_documentation([modification])
                    
                    # Save the successful issue resolution
                    self.identified_issues.append(issue)
                    self._save_issues()
        
        print("\n" + "="*60)
        print("MODIFICATION CYCLE COMPLETE")
        print(f"Successful modifications: {sum(1 for m in self.modification_history if m.success)}")
        print(f"Total modifications attempted: {len(self.modification_history)}")


def main():
    """Example usage"""
    
    optimizer = SelfModifyingOptimizer()
    
    # Example validation results showing TON can't trade
    validation_results = {
        'TON': {
            'blocked_by_threshold': True,
            'typical_volume': 2e5,  # 200K
            'required_threshold': 1e6  # 1M hardcoded
        }
    }
    
    # Run the self-modifying optimization
    optimizer.run_adaptive_modification(validation_results)


if __name__ == "__main__":
    main()