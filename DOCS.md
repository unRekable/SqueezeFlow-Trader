# Documentation Guide

## Where to Find Information

| Topic | File | Section |
|-------|------|---------|
| **Getting Started** | QUICK_START.md | Full file |
| **Project Overview** | README.md | Full file |
| **What Actually Works** | SYSTEM_TRUTH.md | Full file |
| **Trading Strategy** | STRATEGY_IMPLEMENTATION.md | 5-phase methodology |
| **System Design** | SYSTEM_ARCHITECTURE.md | Components & data flow |
| **AI Guidelines** | CLAUDE.md | Development rules |

## Where to Update Documentation

### When to update each file:

**README.md** - Update when:
- Adding major features
- Changing project description
- Updating performance metrics

**QUICK_START.md** - Update when:
- Setup process changes
- New dependencies added
- Common issues discovered

**SYSTEM_TRUTH.md** - Update when:
- Configuration changes
- Discovering what works/doesn't
- Performance benchmarks change
- Data source changes

**STRATEGY_IMPLEMENTATION.md** - Update when:
- Strategy logic changes
- New indicators added
- Scoring system modified

**SYSTEM_ARCHITECTURE.md** - Update when:
- Architecture changes
- New services added
- Data flow modified

**CLAUDE.md** - Update when:
- Development rules change
- New patterns discovered
- AI interaction improves

## Documentation Principles

1. **One source of truth** - Each topic lives in ONE file
2. **No duplication** - Reference other docs, don't copy
3. **Keep it current** - Update immediately when things change
4. **Test everything** - All commands must work
5. **Be specific** - Use real examples, not placeholders

## Quick Reference

### For Users
Start → QUICK_START.md → README.md → STRATEGY_IMPLEMENTATION.md

### For Developers  
Start → SYSTEM_ARCHITECTURE.md → SYSTEM_TRUTH.md → CLAUDE.md

### For Troubleshooting
Start → SYSTEM_TRUTH.md → QUICK_START.md (Troubleshooting section)

### For AI Assistants
Primary: CLAUDE.md, SYSTEM_TRUTH.md
Reference: This file (DOCS.md) for where to update