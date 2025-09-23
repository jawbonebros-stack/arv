#!/usr/bin/env python3
"""
Demonstration of AI Analysis System for ARVLab
Shows how the feature works with saved descriptors
"""
import json

def create_demo_analysis():
    """Create a demonstration AI analysis result"""
    # Simulate descriptors that were just saved
    descriptors = {
        'colours': [{'content': 'bright red'}],
        'tactile': [{'content': 'smooth'}],
        'energy': [{'content': 'warm'}],
        'smell': [{'content': 'sweet'}],
        'sound': [{'content': 'quiet'}],
        'visual': [{'content': 'round'}]
    }
    
    # Simulate target analysis results
    analysis_results = [
        {
            'target_id': 1,
            'target_name': 'target_001.jpg',
            'overall_match': 87.5,
            'category_matches': {
                'Colors': 92.0,  # High match for "bright red"
                'Tactile': 85.0,  # Good match for "smooth"
                'Energy': 88.0,   # Strong match for "warm"
                'Smell': 82.0,    # Moderate match for "sweet"
                'Sound': 80.0,    # Good match for "quiet"
                'Visual': 95.0    # Excellent match for "round"
            },
            'reasoning': 'Strong correlation found between viewing descriptions and target image. The descriptors "bright red", "round", and "warm energy" align exceptionally well with target characteristics. Particularly notable is the 95% visual match indicating accurate shape perception.'
        },
        {
            'target_id': 2,
            'target_name': 'target_002.jpg', 
            'overall_match': 62.3,
            'category_matches': {
                'Colors': 45.0,
                'Tactile': 70.0,
                'Energy': 55.0,
                'Smell': 68.0,
                'Sound': 75.0,
                'Visual': 60.0
            },
            'reasoning': 'Moderate correlation with some matching elements, particularly in tactile and sound categories. However, color and visual descriptors show significant divergence from target characteristics.'
        }
    ]
    
    # Find best match
    best_match = max(analysis_results, key=lambda x: x['overall_match'])
    
    demo_result = {
        'success': True,
        'results': analysis_results,
        'recommended_target': best_match,
        'analysis_summary': f'AI analysis of your viewing session shows strongest correlation with {best_match["target_name"]} at {best_match["overall_match"]:.1f}% overall match. Your sensory impressions demonstrate particularly strong accuracy in visual and color perception categories.',
        'descriptors_analyzed': len([d for category in descriptors.values() for d in category if d['content'].strip()]),
        'confidence_level': 'High' if best_match['overall_match'] > 80 else 'Moderate'
    }
    
    return demo_result

if __name__ == "__main__":
    result = create_demo_analysis()
    print("AI Analysis Demo Result:")
    print(json.dumps(result, indent=2))