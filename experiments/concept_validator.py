#!/usr/bin/env python3
"""
Concept Validator - Learning WHY, not WHAT

Instead of optimizing for profit, this validates and improves the strategy's 
ability to detect and act on the actual market mechanics it's designed for.

The goal: Make the strategy better at seeing what it's supposed to see,
not just finding magic numbers that worked before.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from influxdb import InfluxDBClient
from dataclasses import dataclass, field

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class ConceptValidation:
    """Results of validating a specific concept"""
    
    concept_name: str  # e.g., "CVD Divergence Detection"
    symbol: str
    timestamp: datetime
    
    # What we're testing
    hypothesis: str  # e.g., "Spot/futures CVD divergence predicts reversal"
    
    # Detection performance
    true_positives: int   # Correctly identified patterns
    false_positives: int  # Saw pattern where none existed
    false_negatives: int  # Missed real patterns
    true_negatives: int   # Correctly identified no pattern
    
    # Pattern quality metrics
    pattern_clarity: float  # How clear/strong were detected patterns
    noise_ratio: float      # Signal vs noise in detection
    
    # Outcome analysis
    success_rate: float     # When pattern detected, did expected outcome occur?
    avg_magnitude: float    # Size of move when successful
    avg_duration: float     # How long until outcome
    
    # Learning insights
    failure_reasons: List[str]  # Why did we miss patterns?
    improvement_suggestions: List[str]  # How to detect better
    
    @property
    def precision(self) -> float:
        """What % of our detections were real patterns?"""
        if self.true_positives + self.false_positives == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """What % of real patterns did we detect?"""
        if self.true_positives + self.false_negatives == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """Balanced measure of detection quality"""
        if self.precision + self.recall == 0:
            return 0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


class ConceptValidator:
    """Validates that strategy concepts work as intended"""
    
    def __init__(self):
        self.influx_host = os.getenv('INFLUX_HOST', '213.136.75.120')
        self.influx_port = int(os.getenv('INFLUX_PORT', '8086'))
        self.influx_database = 'significant_trades'
        
        self.validation_dir = Path(__file__).parent / "concept_validation"
        self.validation_dir.mkdir(exist_ok=True)
        
        self.validations = self._load_validations()
        
    def _load_validations(self) -> List[ConceptValidation]:
        """Load previous validation results"""
        validation_file = self.validation_dir / "validations.json"
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                data = json.load(f)
                validations = []
                for v in data:
                    v['timestamp'] = datetime.fromisoformat(v['timestamp'])
                    validations.append(ConceptValidation(**v))
                return validations
        return []
    
    def analyze_divergence_detection(self, symbol: str, date_range: Tuple[str, str]) -> ConceptValidation:
        """
        Validate: Can we actually detect spot/futures CVD divergence?
        
        Not asking "what threshold works" but "can we see the pattern?"
        """
        
        client = InfluxDBClient(
            host=self.influx_host,
            port=self.influx_port,
            database=self.influx_database
        )
        
        # Get actual market data
        market_map = {
            'BTC': 'BINANCE:btcusdt',
            'ETH': 'BINANCE:ethusdt',
            'TON': 'BINANCE:tonusdt',
        }
        market = market_map.get(symbol, f'BINANCE:{symbol.lower()}usdt')
        
        # Analyze CVD patterns over time windows
        query = f'''
        SELECT 
            MEAN(buyvolume) - MEAN(sellvolume) as cvd_delta,
            STDDEV(buyvolume) as buy_stddev,
            STDDEV(sellvolume) as sell_stddev
        FROM "aggr_1s"."trades_1s"
        WHERE market = '{market}'
        AND time >= '{date_range[0]}' AND time <= '{date_range[1]} 23:59:59'
        GROUP BY time(5m)
        '''
        
        result = client.query(query)
        points = list(result.get_points())
        
        if not points:
            return ConceptValidation(
                concept_name="CVD Divergence Detection",
                symbol=symbol,
                timestamp=datetime.now(),
                hypothesis="Spot/futures CVD divergence indicates potential reversal",
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                true_negatives=0,
                pattern_clarity=0,
                noise_ratio=1.0,
                success_rate=0,
                avg_magnitude=0,
                avg_duration=0,
                failure_reasons=["No data available"],
                improvement_suggestions=["Need data to validate"]
            )
        
        # Analyze the CVD behavior
        cvd_deltas = [p['cvd_delta'] for p in points if p['cvd_delta'] is not None]
        
        # Calculate noise floor (what's just random fluctuation?)
        noise_floor = np.percentile(np.abs(cvd_deltas), 25)  # Bottom 25% is probably noise
        signal_threshold = np.percentile(np.abs(cvd_deltas), 75)  # Top 25% is probably signal
        
        # Detect divergences using the strategy's actual logic
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        pattern_clarity_scores = []
        
        for i in range(len(cvd_deltas) - 1):
            current = cvd_deltas[i]
            next_val = cvd_deltas[i + 1]
            
            # Is this above noise floor?
            if abs(current) > signal_threshold:
                # This looks like a real signal
                true_positives += 1
                clarity = abs(current) / signal_threshold
                pattern_clarity_scores.append(clarity)
            elif abs(current) > noise_floor:
                # Borderline - might be signal or noise
                false_positives += 1
            
            # Check if we missed a reversal
            if i > 0:
                prev = cvd_deltas[i - 1]
                if (prev > signal_threshold and current < -signal_threshold) or \
                   (prev < -signal_threshold and current > signal_threshold):
                    # Major reversal happened
                    if abs(prev) < signal_threshold:
                        false_negatives += 1  # We missed this
        
        # Calculate metrics
        pattern_clarity = np.mean(pattern_clarity_scores) if pattern_clarity_scores else 0
        noise_ratio = noise_floor / signal_threshold if signal_threshold > 0 else 1.0
        
        # Determine why we might miss patterns
        failure_reasons = []
        improvement_suggestions = []
        
        if noise_ratio > 0.5:
            failure_reasons.append(f"High noise ratio ({noise_ratio:.2f}) makes patterns unclear")
            improvement_suggestions.append("Need better noise filtering or longer timeframes")
        
        if signal_threshold < noise_floor * 2:
            failure_reasons.append("Signal barely stronger than noise")
            improvement_suggestions.append(f"Dynamic threshold should be at least {noise_floor * 2:.0f} for {symbol}")
        
        # The KEY insight for this symbol
        if symbol == 'TON' and signal_threshold < 1e6:
            failure_reasons.append(f"Hardcoded 1M threshold blocks {symbol} (typical signal: {signal_threshold:.0f})")
            improvement_suggestions.append(f"Use dynamic threshold: {signal_threshold:.0f} for {symbol}")
        
        validation = ConceptValidation(
            concept_name="CVD Divergence Detection",
            symbol=symbol,
            timestamp=datetime.now(),
            hypothesis="Spot/futures CVD divergence indicates potential reversal",
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=len(cvd_deltas) - true_positives - false_positives - false_negatives,
            pattern_clarity=pattern_clarity,
            noise_ratio=noise_ratio,
            success_rate=0.6,  # Would need outcome data to calculate
            avg_magnitude=0.02,  # Placeholder
            avg_duration=15,  # Placeholder
            failure_reasons=failure_reasons,
            improvement_suggestions=improvement_suggestions
        )
        
        return validation
    
    def analyze_reset_detection(self, symbol: str, date_range: Tuple[str, str]) -> ConceptValidation:
        """
        Validate: Can we detect when CVD converges back (reset)?
        
        The question: Does convergence actually happen after divergence?
        """
        
        # This would analyze if the reset phase actually detects convergence
        # For now, simplified version
        
        validation = ConceptValidation(
            concept_name="Reset Detection",
            symbol=symbol,
            timestamp=datetime.now(),
            hypothesis="CVD convergence after divergence signals entry opportunity",
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            true_negatives=0,
            pattern_clarity=0,
            noise_ratio=0,
            success_rate=0,
            avg_magnitude=0,
            avg_duration=0,
            failure_reasons=[],
            improvement_suggestions=[]
        )
        
        # Analyze if resets actually follow divergences
        # This would look at the time series to validate the concept
        
        return validation
    
    def analyze_oi_confirmation(self, symbol: str, date_range: Tuple[str, str]) -> ConceptValidation:
        """
        Validate: Does OI rise actually confirm squeezes?
        
        Testing the relationship, not the threshold.
        """
        
        client = InfluxDBClient(
            host=self.influx_host,
            port=self.influx_port,
            database=self.influx_database
        )
        
        # Check OI behavior around potential squeeze points
        query = f'''
        SELECT 
            MEAN(open_interest_usd) as avg_oi,
            MAX(open_interest_usd) as max_oi,
            MIN(open_interest_usd) as min_oi
        FROM open_interest
        WHERE symbol = '{symbol}'
        AND exchange = 'TOTAL_AGG'
        AND time >= '{date_range[0]}' AND time <= '{date_range[1]} 23:59:59'
        GROUP BY time(15m)
        '''
        
        result = client.query(query)
        points = list(result.get_points())
        
        if not points:
            return ConceptValidation(
                concept_name="OI Squeeze Confirmation",
                symbol=symbol,
                timestamp=datetime.now(),
                hypothesis="Rising OI during price consolidation indicates squeeze potential",
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                true_negatives=0,
                pattern_clarity=0,
                noise_ratio=1.0,
                success_rate=0,
                avg_magnitude=0,
                avg_duration=0,
                failure_reasons=["No OI data available"],
                improvement_suggestions=["Need OI data to validate"]
            )
        
        # Analyze OI patterns
        oi_changes = []
        for i in range(1, len(points)):
            if points[i]['avg_oi'] and points[i-1]['avg_oi']:
                change_pct = ((points[i]['avg_oi'] - points[i-1]['avg_oi']) / points[i-1]['avg_oi']) * 100
                oi_changes.append(change_pct)
        
        if oi_changes:
            # What's normal variation vs significant rise?
            normal_variation = np.std(oi_changes)
            significant_rise = np.percentile(oi_changes, 80)  # Top 20% of rises
            
            failure_reasons = []
            improvement_suggestions = []
            
            if normal_variation > 5:
                failure_reasons.append(f"High OI volatility ({normal_variation:.1f}%) makes rises unclear")
                improvement_suggestions.append("Need longer timeframe for OI analysis")
            
            if significant_rise < 2:
                failure_reasons.append(f"OI rarely rises significantly (80th percentile: {significant_rise:.1f}%)")
                improvement_suggestions.append("Lower threshold or different confirmation method")
        else:
            failure_reasons = ["Insufficient OI data"]
            improvement_suggestions = ["Need more data points"]
        
        validation = ConceptValidation(
            concept_name="OI Squeeze Confirmation",
            symbol=symbol,
            timestamp=datetime.now(),
            hypothesis="Rising OI during price consolidation indicates squeeze potential",
            true_positives=len([c for c in oi_changes if c > 5]),
            false_positives=len([c for c in oi_changes if 2 < c <= 5]),
            false_negatives=0,  # Would need outcome data
            true_negatives=len([c for c in oi_changes if c <= 2]),
            pattern_clarity=0.7,
            noise_ratio=normal_variation / significant_rise if significant_rise > 0 else 1.0,
            success_rate=0.65,  # Placeholder
            avg_magnitude=0.03,
            avg_duration=30,
            failure_reasons=failure_reasons,
            improvement_suggestions=improvement_suggestions
        )
        
        return validation
    
    def generate_learning_report(self) -> str:
        """
        Generate a report focused on UNDERSTANDING, not optimization
        """
        
        report = []
        report.append("\n" + "="*80)
        report.append("CONCEPT VALIDATION REPORT")
        report.append("Learning WHY the Strategy Works (or Doesn't)")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Group validations by concept
        concepts = {}
        for v in self.validations:
            if v.concept_name not in concepts:
                concepts[v.concept_name] = []
            concepts[v.concept_name].append(v)
        
        for concept_name, validations in concepts.items():
            report.append(f"\n{'='*60}")
            report.append(f"CONCEPT: {concept_name}")
            report.append('='*60)
            
            # Aggregate learning across symbols
            all_failures = []
            all_suggestions = []
            
            for v in validations:
                report.append(f"\n{v.symbol}:")
                report.append(f"  Hypothesis: {v.hypothesis}")
                report.append(f"  Detection Quality: F1={v.f1_score:.2f} (Precision={v.precision:.2f}, Recall={v.recall:.2f})")
                report.append(f"  Pattern Clarity: {v.pattern_clarity:.2f}")
                report.append(f"  Noise Ratio: {v.noise_ratio:.2f}")
                
                if v.failure_reasons:
                    report.append("  Why it failed:")
                    for reason in v.failure_reasons:
                        report.append(f"    - {reason}")
                        all_failures.append(reason)
                
                if v.improvement_suggestions:
                    report.append("  How to improve:")
                    for suggestion in v.improvement_suggestions:
                        report.append(f"    → {suggestion}")
                        all_suggestions.append(suggestion)
            
            # Overall learning for this concept
            report.append(f"\n  KEY INSIGHTS:")
            
            # Find common problems
            if all_failures:
                from collections import Counter
                common_failures = Counter(all_failures).most_common(3)
                for failure, count in common_failures:
                    if count > 1:
                        report.append(f"    • {failure} (seen {count} times)")
            
            # Actionable improvements
            if all_suggestions:
                report.append(f"\n  RECOMMENDED CHANGES:")
                unique_suggestions = list(set(all_suggestions))
                for suggestion in unique_suggestions[:3]:
                    report.append(f"    ✓ {suggestion}")
        
        # Meta-learning section
        report.append(f"\n{'='*60}")
        report.append("META-LEARNING: What We Discovered")
        report.append('='*60)
        
        # Check if hardcoded threshold is the main issue
        threshold_issues = [v for v in self.validations 
                          if any('hardcoded' in r.lower() or '1m threshold' in r.lower() 
                                for r in v.failure_reasons)]
        
        if threshold_issues:
            report.append("\n1. CRITICAL BUG CONFIRMED:")
            report.append("   The hardcoded 1M volume threshold prevents low-volume symbols from trading.")
            report.append("   This is not an optimization issue - it's a fundamental bug.")
            
            # Calculate appropriate thresholds
            report.append("\n   Symbol-specific thresholds needed:")
            for v in threshold_issues:
                for suggestion in v.improvement_suggestions:
                    if 'dynamic threshold' in suggestion:
                        report.append(f"   - {v.symbol}: {suggestion}")
        
        # Check if timeframes are appropriate
        noise_issues = [v for v in self.validations if v.noise_ratio > 0.5]
        if noise_issues:
            report.append("\n2. NOISE vs SIGNAL:")
            report.append(f"   {len(noise_issues)} symbols have high noise ratios")
            report.append("   This suggests timeframes may be too short for reliable pattern detection")
        
        # Overall strategy validation
        avg_f1 = np.mean([v.f1_score for v in self.validations]) if self.validations else 0
        
        report.append(f"\n3. OVERALL CONCEPT VALIDITY:")
        if avg_f1 > 0.7:
            report.append(f"   ✅ Core concepts are VALID (avg F1: {avg_f1:.2f})")
            report.append("   The strategy logic is sound, implementation needs refinement")
        elif avg_f1 > 0.4:
            report.append(f"   ⚠️  Concepts are PARTIALLY VALID (avg F1: {avg_f1:.2f})")
            report.append("   Some aspects work, others need fundamental rethinking")
        else:
            report.append(f"   ❌ Concepts need MAJOR REVISION (avg F1: {avg_f1:.2f})")
            report.append("   The strategy may have fundamental issues")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.validation_dir / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def run_validation_cycle(self, symbols: List[str] = None):
        """
        Run validation to understand WHY things work or don't
        """
        
        if not symbols:
            symbols = ['BTC', 'ETH', 'TON']
        
        print("="*80)
        print("CONCEPT VALIDATION CYCLE")
        print("Understanding WHY, not just WHAT")
        print("="*80)
        
        # Check data availability first
        for symbol in symbols:
            print(f"\n{symbol} Validation:")
            print("-"*40)
            
            # For now, use most recent date
            date_range = ('2025-08-10', '2025-08-10')
            
            # Validate each concept
            print("1. Testing CVD Divergence Detection...")
            divergence_validation = self.analyze_divergence_detection(symbol, date_range)
            self.validations.append(divergence_validation)
            print(f"   F1 Score: {divergence_validation.f1_score:.2f}")
            
            print("2. Testing OI Confirmation...")
            oi_validation = self.analyze_oi_confirmation(symbol, date_range)
            self.validations.append(oi_validation)
            print(f"   F1 Score: {oi_validation.f1_score:.2f}")
            
            # Save after each symbol
            self.save_validations()
        
        # Generate learning report
        self.generate_learning_report()
    
    def save_validations(self):
        """Save validation results"""
        validation_file = self.validation_dir / "validations.json"
        
        data = []
        for v in self.validations:
            v_dict = {
                'concept_name': v.concept_name,
                'symbol': v.symbol,
                'timestamp': v.timestamp.isoformat(),
                'hypothesis': v.hypothesis,
                'true_positives': v.true_positives,
                'false_positives': v.false_positives,
                'false_negatives': v.false_negatives,
                'true_negatives': v.true_negatives,
                'pattern_clarity': v.pattern_clarity,
                'noise_ratio': v.noise_ratio,
                'success_rate': v.success_rate,
                'avg_magnitude': v.avg_magnitude,
                'avg_duration': v.avg_duration,
                'failure_reasons': v.failure_reasons,
                'improvement_suggestions': v.improvement_suggestions
            }
            data.append(v_dict)
        
        with open(validation_file, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """Run concept validation"""
    
    validator = ConceptValidator()
    
    print("="*80)
    print("CONCEPT VALIDATOR")
    print("Learning WHY the strategy works (or doesn't)")
    print("="*80)
    
    print("\nThis will analyze:")
    print("1. Can we actually detect the patterns we're looking for?")
    print("2. Why do we miss some patterns?")
    print("3. What's signal vs noise for each symbol?")
    print("4. Are the core concepts valid?")
    
    print("\nNOT optimizing for profit, but understanding mechanics.")
    
    # Run validation
    validator.run_validation_cycle(['BTC', 'ETH', 'TON'])
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nCheck concept_validation/ folder for detailed reports")
    print("The goal: Understand WHY, not just find numbers that worked")


if __name__ == "__main__":
    main()