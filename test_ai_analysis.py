#!/usr/bin/env python3
"""
Test script to demonstrate AI analysis functionality
"""
import os
from ai_analysis import AIAnalysisEngine

def test_ai_analysis():
    """Test the AI analysis feature with a real trial"""
    print("Testing AI Analysis System...")
    
    # Initialize the AI analysis engine
    engine = AIAnalysisEngine()
    
    # Test with trial ID 5 and user ID 1 (from earlier testing)
    result = engine.analyze_viewing_session(trial_id=5, user_id=1)
    
    if result:
        print("✓ AI Analysis successful!")
        print(f"Recommended target: {result.recommended_target.target_name}")
        print(f"Overall match: {result.recommended_target.overall_match:.1f}%")
        print(f"Summary: {result.analysis_summary}")
        
        print("\nDetailed results:")
        for i, target_result in enumerate(result.results):
            print(f"  Target {i+1}: {target_result.target_name} - {target_result.overall_match:.1f}%")
    else:
        print("✗ AI Analysis failed")

if __name__ == "__main__":
    test_ai_analysis()