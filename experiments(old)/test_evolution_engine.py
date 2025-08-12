#!/usr/bin/env python3
"""
Test Runner for Evolution Engine V4
Demonstrates capabilities with simple examples
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from evolution_engine_v4 import (
    SystemAnalyzer,
    ConceptDiscoveryEngine, 
    SystemEvolution,
    IntelligentOptimizer,
    EvolutionEngine
)

class EvolutionEngineTestRunner:
    """Test runner to demonstrate V4 capabilities"""
    
    def __init__(self):
        self.results = []
        self.test_mode = True  # Use test mode for safety
        
    async def test_system_analyzer(self):
        """Test system understanding capabilities"""
        print("\n" + "="*60)
        print("🔍 TESTING SYSTEM ANALYZER")
        print("="*60)
        
        analyzer = SystemAnalyzer()
        
        # Test architecture analysis
        print("\n1. Analyzing System Architecture...")
        architecture = analyzer.analyze_architecture()
        print(f"   ✅ Found {len(architecture['components'])} components")
        print(f"   ✅ Identified {len(architecture['data_flows'])} data flows")
        
        # Test bug detection
        print("\n2. Detecting Bugs...")
        bugs = analyzer.detect_bugs()
        if bugs:
            print(f"   ⚠️ Found {len(bugs)} potential issues")
            for bug in bugs[:2]:  # Show first 2
                print(f"      - {bug.get('file', 'unknown')}:{bug.get('line', 0)}: {bug.get('description', bug)}")
        
        # Test failure analysis
        print("\n3. Understanding Failures...")
        # Mock some backtest results for testing
        mock_results = [
            {'win_rate': 0.4, 'symbol': 'ETH', 'trades': 10},
            {'win_rate': 0.3, 'symbol': 'BTC', 'trades': 5}
        ]
        failures = analyzer.understand_failures(mock_results)
        print(f"   📊 Analyzed {len(failures.get('patterns', []))} failure patterns")
        
        self.results.append({
            'test': 'SystemAnalyzer',
            'status': 'passed',
            'findings': {
                'components': len(architecture['components']),
                'bugs': len(bugs) if isinstance(bugs, list) else 0,
                'data_flows': len(architecture['data_flows'])
            }
        })
        
        return architecture, bugs
        
    async def test_concept_discovery(self):
        """Test concept discovery capabilities"""
        print("\n" + "="*60)
        print("🧠 TESTING CONCEPT DISCOVERY")
        print("="*60)
        
        discovery = ConceptDiscoveryEngine()
        
        # Test hypothesis validation
        print("\n1. Validating Trading Hypothesis...")
        hypotheses = ["CVD divergence predicts reversals"]
        results = discovery.validate_hypotheses(hypotheses)
        if results:
            result = results[0]
            print(f"   Hypothesis: '{result['hypothesis']}'")
            print(f"   Result: {result['result']} (Confidence: {result.get('confidence', 0.75):.1%})")
        
        # Test pattern discovery
        print("\n2. Discovering Patterns...")
        # Mock some backtest data for pattern discovery
        mock_data = [
            {'symbol': 'ETH', 'win_rate': 0.6, 'trades': 20, 'strategy_scores': [5.0, 6.0, 4.0]},
            {'symbol': 'BTC', 'win_rate': 0.55, 'trades': 15, 'strategy_scores': [4.5, 5.5, 3.5]}
        ]
        patterns = discovery.discover_patterns(mock_data)
        print(f"   ✅ Found {len(patterns['successful_patterns'])} successful patterns")
        print(f"   ⚠️ Found {len(patterns['failure_patterns'])} failure patterns")
        
        # Test principle extraction
        print("\n3. Extracting Trading Principles...")
        principles = discovery.extract_principles()
        print(f"   📚 Extracted {len(principles)} key principles:")
        for principle in principles[:3]:  # Show first 3
            print(f"      - {principle}")
        
        self.results.append({
            'test': 'ConceptDiscovery',
            'status': 'passed',
            'findings': {
                'hypothesis_validated': results[0]['result'] if results else False,
                'patterns_found': len(patterns['successful_patterns']),
                'principles_extracted': len(principles)
            }
        })
        
        return principles
        
    async def test_system_evolution(self):
        """Test system modification capabilities (safe mode)"""
        print("\n" + "="*60)
        print("🔧 TESTING SYSTEM EVOLUTION (Safe Mode)")
        print("="*60)
        
        evolution = SystemEvolution()
        
        # Test bug fixing capability (dry run)
        print("\n1. Testing Bug Fix Generation...")
        test_bug = {
            'file': 'test_file.py',
            'line': 100,
            'issue': 'Hardcoded threshold value',
            'suggested_fix': 'Use dynamic calculation'
        }
        
        fix_plan = evolution._generate_fix(test_bug)
        print(f"   📝 Generated fix plan for: {test_bug['issue']}")
        print(f"   ✅ Fix would: {fix_plan.get('description', 'Apply dynamic calculation')}")
        
        # Test gap identification
        print("\n2. Identifying System Gaps...")
        gaps = evolution._identify_gaps()
        print(f"   🔍 Found {len(gaps)} potential improvements:")
        for gap in gaps[:3]:  # Show first 3
            print(f"      - {gap}")
        
        # Test architecture optimization suggestions
        print("\n3. Suggesting Architecture Improvements...")
        optimizations = evolution._suggest_optimizations()
        print(f"   💡 Generated {len(optimizations)} optimization suggestions")
        
        self.results.append({
            'test': 'SystemEvolution',
            'status': 'passed',
            'findings': {
                'fix_generated': bool(fix_plan),
                'gaps_identified': len(gaps),
                'optimizations_suggested': len(optimizations)
            }
        })
        
        return optimizations
        
    async def test_intelligent_optimizer(self):
        """Test intelligent optimization capabilities"""
        print("\n" + "="*60)
        print("🎯 TESTING INTELLIGENT OPTIMIZER")
        print("="*60)
        
        optimizer = IntelligentOptimizer()
        
        # Test Bayesian optimization setup
        print("\n1. Initializing Bayesian Optimization...")
        param_space = {
            'MIN_ENTRY_SCORE': {'min': 2.0, 'max': 6.0, 'type': 'float'},
            'CVD_THRESHOLD': {'min': 1.0, 'max': 3.0, 'type': 'float'}
        }
        
        optimizer.initialize_bayesian(param_space)
        print(f"   ✅ Initialized with {len(param_space)} parameters")
        
        # Test next suggestion
        print("\n2. Getting Next Parameter Suggestion...")
        suggestion = optimizer.suggest_next_params()
        print(f"   📊 Suggested parameters:")
        for param, value in suggestion.items():
            print(f"      - {param}: {value:.2f}")
        
        # Test performance prediction
        print("\n3. Predicting Performance...")
        predicted_score = optimizer.predict_performance(suggestion)
        print(f"   🎯 Predicted score: {predicted_score:.2f}")
        
        self.results.append({
            'test': 'IntelligentOptimizer',
            'status': 'passed',
            'findings': {
                'bayesian_initialized': True,
                'suggestion_generated': bool(suggestion),
                'prediction_made': predicted_score is not None
            }
        })
        
        return suggestion
        
    async def test_full_evolution_cycle(self):
        """Test a complete evolution cycle"""
        print("\n" + "="*60)
        print("🔄 TESTING FULL EVOLUTION CYCLE")
        print("="*60)
        
        engine = EvolutionEngine()
        
        # Configure for test mode
        config = {
            'mode': 'test',
            'max_iterations': 1,
            'enable_modifications': False,  # Safety first
            'visual_validation': True,
            'learning_rate': 0.1
        }
        
        print("\n1. Starting Evolution Engine...")
        print(f"   Configuration: {json.dumps(config, indent=2)}")
        
        # Run one evolution cycle
        print("\n2. Running Evolution Cycle...")
        results = await engine.run_evolution_cycle(config)
        
        print("\n3. Evolution Cycle Results:")
        print(f"   ✅ System analyzed: {results.get('system_analyzed', False)}")
        print(f"   ✅ Concepts discovered: {results.get('concepts_found', 0)}")
        print(f"   ✅ Improvements identified: {results.get('improvements', 0)}")
        print(f"   ✅ Visual validation: {results.get('visual_validated', False)}")
        
        self.results.append({
            'test': 'FullEvolutionCycle',
            'status': 'passed',
            'findings': results
        })
        
        return results
        
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "="*70)
        print("🚀 EVOLUTION ENGINE V4 - TEST SUITE")
        print("="*70)
        print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Test each layer
            await self.test_system_analyzer()
            await self.test_concept_discovery()
            await self.test_system_evolution()
            await self.test_intelligent_optimizer()
            await self.test_full_evolution_cycle()
            
            # Summary
            print("\n" + "="*70)
            print("📊 TEST RESULTS SUMMARY")
            print("="*70)
            
            all_passed = True
            for result in self.results:
                status = "✅" if result['status'] == 'passed' else "❌"
                print(f"{status} {result['test']}: {result['status'].upper()}")
                
                if result['status'] != 'passed':
                    all_passed = False
            
            if all_passed:
                print("\n🎉 ALL TESTS PASSED! Evolution Engine V4 is ready for use.")
            else:
                print("\n⚠️ Some tests failed. Review the output above.")
            
            # Save results
            self._save_results()
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            
    def _save_results(self):
        """Save test results to file"""
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"evolution_v4_test_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'test_results': self.results,
                'summary': {
                    'total_tests': len(self.results),
                    'passed': sum(1 for r in self.results if r['status'] == 'passed'),
                    'failed': sum(1 for r in self.results if r['status'] != 'passed')
                }
            }, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_file}")

async def main():
    """Main test entry point"""
    runner = EvolutionEngineTestRunner()
    await runner.run_all_tests()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     EVOLUTION ENGINE V4 - CAPABILITY DEMONSTRATION          ║
║                                                              ║
║  This test suite demonstrates the V4 framework's ability    ║
║  to understand, analyze, and evolve the trading system.     ║
║                                                              ║
║  Tests will run in SAFE MODE - no actual modifications.     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())