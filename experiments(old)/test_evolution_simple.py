#!/usr/bin/env python3
"""
Simple test to verify Evolution Engine V4 basic functionality
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.evolution_engine_v4 import (
    SystemAnalyzer,
    ConceptDiscoveryEngine,
    SystemEvolution,
    IntelligentOptimizer,
    EvolutionEngine
)

def test_system_analyzer():
    """Test basic system analysis"""
    print("\n🔍 Testing System Analyzer...")
    analyzer = SystemAnalyzer()
    
    # Test architecture analysis
    architecture = analyzer.analyze_architecture()
    print(f"✅ Found {len(architecture['components'])} components")
    print(f"✅ Identified {len(architecture['data_flows'])} data flows")
    
    # Test bug detection
    bugs = analyzer.detect_bugs()
    print(f"✅ Detected {len(bugs)} potential issues")
    
    return True

def test_concept_discovery():
    """Test concept discovery basics"""
    print("\n🧠 Testing Concept Discovery...")
    discovery = ConceptDiscoveryEngine()
    
    # Test with mock data
    mock_data = [
        {'symbol': 'ETH', 'win_rate': 0.6, 'trades': 20},
        {'symbol': 'BTC', 'win_rate': 0.55, 'trades': 15}
    ]
    
    # Test hypothesis validation
    hypotheses = ["High volume leads to better trades"]
    results = discovery.validate_hypotheses(hypotheses, mock_data)
    print(f"✅ Validated {len(results)} hypotheses")
    
    # Test pattern discovery
    patterns = discovery.discover_patterns(mock_data)
    print(f"✅ Found {len(patterns['successful_patterns'])} successful patterns")
    
    # Test principle extraction
    principles = discovery.extract_principles()
    print(f"✅ Extracted {len(principles)} principles")
    
    return True

def test_system_evolution():
    """Test system evolution capabilities"""
    print("\n🔧 Testing System Evolution...")
    evolution = SystemEvolution()
    
    # Just test that methods exist and don't crash
    print("✅ System Evolution initialized")
    
    # Test internal methods exist
    assert hasattr(evolution, '_generate_fix')
    assert hasattr(evolution, '_identify_gaps')
    assert hasattr(evolution, '_suggest_optimizations')
    print("✅ All evolution methods available")
    
    return True

def test_intelligent_optimizer():
    """Test intelligent optimizer"""
    print("\n🎯 Testing Intelligent Optimizer...")
    optimizer = IntelligentOptimizer()
    
    # Test parameter space setup
    param_space = {
        'MIN_ENTRY_SCORE': {'min': 2.0, 'max': 6.0, 'type': 'float'}
    }
    
    optimizer.initialize_bayesian(param_space)
    print("✅ Bayesian optimization initialized")
    
    # Test suggestion
    suggestion = optimizer.suggest_next_params()
    print(f"✅ Generated parameter suggestion: {suggestion}")
    
    # Test prediction
    score = optimizer.predict_performance(suggestion)
    print(f"✅ Predicted performance: {score:.2f}")
    
    return True

def test_evolution_engine():
    """Test main evolution engine"""
    print("\n🔄 Testing Evolution Engine...")
    engine = EvolutionEngine()
    
    print("✅ Evolution Engine initialized")
    
    # Test that all components are present
    assert hasattr(engine, 'analyzer')
    assert hasattr(engine, 'discovery')
    assert hasattr(engine, 'evolution')
    assert hasattr(engine, 'optimizer')
    assert hasattr(engine, 'visual')
    assert hasattr(engine, 'learning')
    print("✅ All engine components present")
    
    return True

def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         EVOLUTION ENGINE V4 - SIMPLE TEST SUITE             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("System Analyzer", test_system_analyzer),
        ("Concept Discovery", test_concept_discovery),
        ("System Evolution", test_system_evolution),
        ("Intelligent Optimizer", test_intelligent_optimizer),
        ("Evolution Engine", test_evolution_engine)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASSED"))
        except Exception as e:
            results.append((name, f"❌ FAILED: {e}"))
            print(f"❌ {name} failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, status in results:
        print(f"{name}: {status}")
    
    passed = sum(1 for _, s in results if "PASSED" in s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Evolution Engine V4 is ready.")
    else:
        print("\n⚠️ Some tests failed. Review the output above.")

if __name__ == "__main__":
    main()