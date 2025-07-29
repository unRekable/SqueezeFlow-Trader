"""
Trading Strategies
Collection of SqueezeFlow trading strategy implementations

Available Strategies:
- production_enhanced_strategy: Production-ready enhanced SqueezeFlow strategy
- simple_squeeze_strategy: Basic squeeze detection strategy
- working_squeeze_strategy: Working implementation with momentum alignment
- enhanced_squeezeflow_strategy: Advanced multi-timeframe strategy
- debug_strategy: Debug and testing strategy
- squeezeflow_strategy: Original base SqueezeFlow implementation
- working_strategy: Alternative working implementation

Strategy Discovery:
All strategies are automatically discovered and can be loaded by name
using the load_strategy() function from core.strategy.
"""

import os
import importlib
from typing import List, Dict, Any

def get_available_strategies() -> List[str]:
    """Get list of available strategy names"""
    strategy_dir = os.path.dirname(__file__)
    strategies = []
    
    for file in os.listdir(strategy_dir):
        if file.endswith('_strategy.py') and not file.startswith('__'):
            strategy_name = file.replace('.py', '')
            strategies.append(strategy_name)
    
    return sorted(strategies)

def get_strategy_info() -> Dict[str, Any]:
    """Get detailed information about available strategies"""
    strategies = get_available_strategies()
    info = {}
    
    for strategy_name in strategies:
        try:
            module = importlib.import_module(f'.{strategy_name}', package=__name__)
            info[strategy_name] = {
                'module': module,
                'doc': getattr(module, '__doc__', 'No description available'),
                'classes': [name for name in dir(module) if name.endswith('Strategy')]
            }
        except ImportError as e:
            info[strategy_name] = {'error': str(e)}
    
    return info

__all__ = ["get_available_strategies", "get_strategy_info"]