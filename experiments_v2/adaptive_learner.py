#!/usr/bin/env python3
"""
Adaptive Learning System

This maintains continuity across sessions and adapts to new findings.
It's designed to be "picked up" by any Claude session and continue learning.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LearningEntry:
    """A single learning from validation"""
    timestamp: str
    symbol: str
    concept: str
    finding: str
    confidence: float  # 0-1, how sure are we?
    supersedes: Optional[str] = None  # ID of previous learning this updates
    
    @property
    def id(self) -> str:
        return f"{self.symbol}_{self.concept}_{self.timestamp}"


@dataclass 
class OpenQuestion:
    """Questions that need investigation"""
    question: str
    priority: int  # 1-5, higher = more important
    context: str
    suggested_approach: str
    created: str
    status: str = "open"  # open, investigating, answered
    answer: Optional[str] = None


class AdaptiveLearner:
    """Maintains learning state across sessions"""
    
    def __init__(self):
        self.learning_dir = Path(__file__).parent / "adaptive_learning"
        self.learning_dir.mkdir(exist_ok=True)
        
        # Core knowledge files
        self.journal_file = self.learning_dir / "learning_journal.json"
        self.questions_file = self.learning_dir / "open_questions.json"
        self.principles_file = self.learning_dir / "discovered_principles.json"
        self.next_steps_file = self.learning_dir / "next_steps.json"
        
        # Load existing knowledge
        self.journal = self._load_journal()
        self.questions = self._load_questions()
        self.principles = self._load_principles()
        self.next_steps = self._load_next_steps()
    
    def _load_journal(self) -> List[LearningEntry]:
        """Load learning history"""
        if self.journal_file.exists():
            with open(self.journal_file, 'r') as f:
                data = json.load(f)
                return [LearningEntry(**entry) for entry in data]
        return []
    
    def _load_questions(self) -> List[OpenQuestion]:
        """Load open questions"""
        if self.questions_file.exists():
            with open(self.questions_file, 'r') as f:
                data = json.load(f)
                return [OpenQuestion(**q) for q in data]
        return []
    
    def _load_principles(self) -> Dict[str, Any]:
        """Load discovered principles"""
        if self.principles_file.exists():
            with open(self.principles_file, 'r') as f:
                return json.load(f)
        return {
            "universal": {},  # Principles that apply to all symbols
            "symbol_specific": {},  # Symbol-specific characteristics
            "conditional": {}  # Principles that depend on conditions
        }
    
    def _load_next_steps(self) -> List[Dict]:
        """Load suggested next steps"""
        if self.next_steps_file.exists():
            with open(self.next_steps_file, 'r') as f:
                return json.load(f)
        return []
    
    def record_learning(self, symbol: str, concept: str, finding: str, 
                       confidence: float, supersedes: Optional[str] = None):
        """Record a new learning"""
        
        entry = LearningEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            concept=concept,
            finding=finding,
            confidence=confidence,
            supersedes=supersedes
        )
        
        self.journal.append(entry)
        
        # Check if this updates our principles
        self._update_principles(entry)
        
        # Generate new questions if needed
        self._generate_questions(entry)
        
        self.save_all()
        
        return entry
    
    def _update_principles(self, entry: LearningEntry):
        """Update principles based on new learning"""
        
        # High confidence findings might establish principles
        if entry.confidence > 0.8:
            # Check if this is universal or symbol-specific
            similar_findings = [e for e in self.journal 
                              if e.concept == entry.concept and e.finding == entry.finding]
            
            if len(similar_findings) >= 3:  # Pattern seen 3+ times
                # This might be a universal principle
                self.principles["universal"][entry.concept] = {
                    "principle": entry.finding,
                    "confidence": entry.confidence,
                    "evidence_count": len(similar_findings)
                }
            else:
                # Symbol-specific learning
                if entry.symbol not in self.principles["symbol_specific"]:
                    self.principles["symbol_specific"][entry.symbol] = {}
                
                self.principles["symbol_specific"][entry.symbol][entry.concept] = {
                    "finding": entry.finding,
                    "confidence": entry.confidence
                }
    
    def _generate_questions(self, entry: LearningEntry):
        """Generate new questions based on learning"""
        
        # Low confidence findings generate questions
        if entry.confidence < 0.5:
            question = OpenQuestion(
                question=f"Why is {entry.concept} unclear for {entry.symbol}?",
                priority=3,
                context=entry.finding,
                suggested_approach="Run deeper analysis with more data",
                created=datetime.now().isoformat()
            )
            self.questions.append(question)
        
        # Contradictions generate high-priority questions
        contradictions = [e for e in self.journal 
                         if e.symbol == entry.symbol 
                         and e.concept == entry.concept
                         and e.finding != entry.finding]
        
        if contradictions:
            question = OpenQuestion(
                question=f"Why does {entry.concept} behave differently at different times for {entry.symbol}?",
                priority=5,
                context=f"New: {entry.finding}, Previous: {contradictions[0].finding}",
                suggested_approach="Investigate market conditions during both periods",
                created=datetime.now().isoformat()
            )
            self.questions.append(question)
    
    def get_next_investigation(self) -> Optional[Dict]:
        """What should we investigate next?"""
        
        # Priority 1: Unanswered high-priority questions
        open_questions = [q for q in self.questions if q.status == "open"]
        if open_questions:
            # Sort by priority
            open_questions.sort(key=lambda q: q.priority, reverse=True)
            top_question = open_questions[0]
            
            return {
                "type": "question",
                "question": top_question.question,
                "approach": top_question.suggested_approach,
                "context": top_question.context
            }
        
        # Priority 2: Validate uncertain principles
        uncertain = [(k, v) for k, v in self.principles["universal"].items() 
                    if v["confidence"] < 0.7]
        
        if uncertain:
            concept, principle = uncertain[0]
            return {
                "type": "validation",
                "concept": concept,
                "principle": principle["principle"],
                "goal": "Increase confidence through more testing"
            }
        
        # Priority 3: Test symbol coverage
        tested_symbols = set(e.symbol for e in self.journal)
        all_symbols = ['BTC', 'ETH', 'TON', 'AVAX', 'SOL']
        untested = [s for s in all_symbols if s not in tested_symbols]
        
        if untested:
            return {
                "type": "exploration",
                "symbol": untested[0],
                "goal": "Understand characteristics of new symbol"
            }
        
        return None
    
    def generate_status_report(self) -> str:
        """Generate a status report for the next Claude session"""
        
        report = []
        report.append("="*60)
        report.append("ADAPTIVE LEARNING STATUS")
        report.append(f"Last Updated: {datetime.now().isoformat()}")
        report.append("="*60)
        
        # Current understanding
        report.append("\n📚 WHAT WE'VE LEARNED:")
        report.append("-"*40)
        
        if self.principles["universal"]:
            report.append("\nUniversal Principles:")
            for concept, principle in self.principles["universal"].items():
                report.append(f"  • {concept}: {principle['principle']} (confidence: {principle['confidence']:.0%})")
        
        if self.principles["symbol_specific"]:
            report.append("\nSymbol-Specific Findings:")
            for symbol, findings in self.principles["symbol_specific"].items():
                report.append(f"\n  {symbol}:")
                for concept, finding in findings.items():
                    report.append(f"    - {concept}: {finding['finding']}")
        
        # Open questions
        open_q = [q for q in self.questions if q.status == "open"]
        if open_q:
            report.append("\n❓ OPEN QUESTIONS:")
            report.append("-"*40)
            for q in open_q[:5]:  # Top 5
                report.append(f"  Priority {q.priority}: {q.question}")
        
        # Next steps
        next_investigation = self.get_next_investigation()
        if next_investigation:
            report.append("\n🎯 RECOMMENDED NEXT STEP:")
            report.append("-"*40)
            report.append(f"  Type: {next_investigation['type']}")
            if 'question' in next_investigation:
                report.append(f"  Question: {next_investigation['question']}")
            report.append(f"  Approach: {next_investigation.get('approach', next_investigation.get('goal'))}")
        
        # Progress metrics
        report.append("\n📊 PROGRESS METRICS:")
        report.append("-"*40)
        report.append(f"  Total learnings: {len(self.journal)}")
        report.append(f"  Universal principles: {len(self.principles['universal'])}")
        report.append(f"  Open questions: {len(open_q)}")
        report.append(f"  Symbols analyzed: {len(set(e.symbol for e in self.journal))}")
        
        # Instructions for next session
        report.append("\n💡 TO CONTINUE:")
        report.append("-"*40)
        report.append("1. Run: python3 experiments/adaptive_learner.py")
        report.append("2. This will show current status and next steps")
        report.append("3. Run suggested investigation")
        report.append("4. Record findings with record_learning()")
        report.append("5. System will adapt and suggest next step")
        
        return "\n".join(report)
    
    def save_all(self):
        """Save all learning state"""
        
        # Save journal
        journal_data = [asdict(e) for e in self.journal]
        with open(self.journal_file, 'w') as f:
            json.dump(journal_data, f, indent=2)
        
        # Save questions
        questions_data = [asdict(q) for q in self.questions]
        with open(self.questions_file, 'w') as f:
            json.dump(questions_data, f, indent=2)
        
        # Save principles
        with open(self.principles_file, 'w') as f:
            json.dump(self.principles, f, indent=2)
        
        # Save next steps
        if self.next_steps:
            with open(self.next_steps_file, 'w') as f:
                json.dump(self.next_steps, f, indent=2)
    
    def answer_question(self, question_text: str, answer: str):
        """Mark a question as answered"""
        for q in self.questions:
            if q.question == question_text:
                q.status = "answered"
                q.answer = answer
                break
        self.save_all()


def main():
    """Entry point for any Claude session"""
    
    learner = AdaptiveLearner()
    
    print("="*60)
    print("ADAPTIVE LEARNING SYSTEM")
    print("Continuous learning across sessions")
    print("="*60)
    
    # Show current status
    print(learner.generate_status_report())
    
    # Check what to do next
    next_step = learner.get_next_investigation()
    
    if next_step:
        print("\n" + "="*60)
        print("READY TO CONTINUE LEARNING")
        print("="*60)
        print(f"\nNext investigation: {next_step}")
        
        print("\nTo record new findings:")
        print("learner.record_learning(symbol='TON', concept='CVD_threshold',")
        print("                        finding='Needs 150K in calm markets',")
        print("                        confidence=0.8)")
    else:
        print("\nNo pending investigations. Consider:")
        print("1. Testing on new date ranges")
        print("2. Testing during different market conditions")
        print("3. Validating principles with forward testing")


if __name__ == "__main__":
    main()