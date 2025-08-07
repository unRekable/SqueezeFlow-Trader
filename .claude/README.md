# Claude Code Configuration - SqueezeFlow Trader

## ✅ Configuration Status

This Claude Code setup follows the official documentation and best practices.

## 📁 Structure

```
.claude/
├── settings.json        # Main configuration with hooks
├── agents/             # Simplified agent set (5 focused agents)
│   ├── strategy.md     # Trading strategy specialist
│   ├── docker.md       # Container deployment
│   ├── testing.md      # Quality assurance
│   ├── database.md     # Data management
│   └── refactor.md     # Code improvement
├── commands/           # Slash commands
│   ├── backtest.md     # Run backtests
│   ├── status.md       # Check service status
│   └── clean.md        # Clean temporary files
└── README.md           # This file
```

## 🎯 Key Features

### Behavioral Rules
1. **User instructions have absolute priority** - Never override explicit user requests
2. **Fix, don't delete** - Diagnose and repair issues without removing functionality
3. **Clean environment** - Automatic cleanup of temporary files

### Hooks (in settings.json)
- **SessionStart**: Display key rules
- **UserPromptSubmit**: Simple timestamp
- **PostToolUse**: Python formatting with Black
- **Stop**: Clean temporary files

### Memory System
- Project memory: `/CLAUDE.md` (main project instructions)
- User memory: `~/.claude/CLAUDE.md` (personal preferences)
- No complex memory hierarchies or enforcement systems

### Agents (Simplified)
From 30+ agents reduced to 5 focused specialists:
- `strategy` - Trading strategy implementation
- `docker` - Container deployment
- `testing` - Quality assurance
- `database` - Data management
- `refactor` - Code improvement (with protection against destructive changes)

### Commands
- `/backtest [timeframe] [balance]` - Run backtests
- `/status [service]` - Check service status
- `/clean` - Clean temporary files

## 🚀 Usage

### Creating Temporary Scripts
Name them with these prefixes for auto-cleanup:
- `test_*.py`
- `debug_*.py`
- `temp_*.py`
- `check_*.py`

They'll be automatically removed after 5 minutes when session ends.

### Running Commands
Use slash commands in chat:
```
/backtest last_week 10000
/status redis
/clean
```

### Using Agents
Agents are automatically selected based on task, or explicitly with Task tool.

## ⚠️ Important Notes

1. **No standalone hook files** - All hooks are in settings.json per Claude docs
2. **Simplified structure** - Removed complex memory systems and excessive agents
3. **User-first approach** - Your instructions always override any rules
4. **Non-destructive** - System will never comment out or remove working code

## 🔧 Maintenance

### To modify hooks
Edit `.claude/settings.json` - all hooks are inline commands.

### To add agents
Create new `.md` file in `agents/` with proper frontmatter.

### To add commands
Create new `.md` file in `commands/` with proper frontmatter.

## 📚 References

- [Claude Code Hooks Documentation](https://docs.anthropic.com/en/docs/claude-code/hooks)
- [Claude Code Memory Documentation](https://docs.anthropic.com/en/docs/claude-code/memory)
- [Claude Code Commands Documentation](https://docs.anthropic.com/en/docs/claude-code/slash-commands)

---

Configuration last updated: 2025-01-07
Follows official Claude Code documentation exactly.