# Architectural Principles - SqueezeFlow Trader

**Purpose:** Define and enforce architectural patterns that prevent cascade failures and maintainability issues

## 🎯 CORE PRINCIPLES

### 1. Single Source of Truth (SSOT)
**Principle:** Every piece of configuration, logic, or data has exactly ONE authoritative source.

**Enforcement:**
- ALL configuration in `/backtest/indicator_config.py`
- ALL components MUST import from there
- NO local config variables
- NO hardcoded values

### 2. Explicit Dependencies (No Hidden Coupling)
**Principle:** Components declare their dependencies explicitly through constructor injection.

### 3. One Implementation Per Function
**Principle:** Each functionality has exactly ONE implementation. No duplicates, variants, or "v2" versions.

### 4. Unidirectional Data Flow
**Principle:** Data flows in one direction through the system. No circular dependencies.

### 5. Fail Fast and Explicitly
**Principle:** Errors surface immediately with clear messages. No silent failures.

See full document for complete patterns and validation rules.
EOF < /dev/null