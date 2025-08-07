---
description: Clean temporary files and artifacts
allowed-tools: Bash, Glob
---

Clean up temporary files and test artifacts from the project.

## What it cleans
- Temporary Python scripts (test_*.py, debug_*.py, temp_*.py)
- Python cache files (__pycache__, *.pyc)
- Test artifacts and coverage reports
- Log files older than 7 days

## Usage
- `/clean` - Run full cleanup
- `/clean --dry-run` - Show what would be deleted

The command will:
1. List files to be removed
2. Ask for confirmation
3. Remove approved files
4. Show cleanup summary