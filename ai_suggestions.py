import json
import os
from typing import Dict, List, Optional, Any
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=30.0
        )
        AI_ENABLED = True
        print("✓ AI suggestions enabled with OpenAI API")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        AI_ENABLED = False
        openai_client = None
else:
    AI_ENABLED = False
    openai_client = None
    print("ℹ AI suggestions disabled (no OpenAI API key)")

def suggest_trial_configuration(title: str, domain: str, description: str = "") -> Dict[str, Any]:
    """
    Generate AI-powered suggestions for trial configuration based on title, domain, and description.
    """
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key"}
    
    try:
        prompt = f"""
You are an expert in Associative Remote Viewing (ARV) prediction tasks. 
Analyze the following trial and provide optimal configuration suggestions:

Title: {title}
Domain: {domain}
Description: {description or "No description provided"}

Provide suggestions in JSON format with these fields:
1. "suggested_title": An improved, more descriptive title if needed
2. "timing_suggestion": Recommended timing considerations 
3. "outcome_suggestions": Specific outcome labels/names based on the trial
4. "target_selection_advice": Guidance on what types of targets work best
5. "description_enhancement": Improved description if the original is lacking
6. "domain_specific_tips": Tips specific to the {domain} domain
7. "success_factors": Key factors that make this type of trial successful

Focus on practical, actionable advice for creating effective ARV trials.
Be specific to the trial context and domain type.
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "You are an expert ARV trial designer. Provide practical, specific suggestions in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        suggestions = json.loads(response.choices[0].message.content)
        return {"suggestions": suggestions, "success": True}
        
    except Exception as e:
        return {"error": f"Failed to generate suggestions: {str(e)}", "success": False}

def suggest_target_selection(domain: str, outcomes: List[str], available_targets: List[Dict]) -> Dict[str, Any]:
    """
    Suggest optimal target selection based on domain, outcomes, and available targets.
    """
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key"}
    
    try:
        # Prepare target information for analysis
        target_info = []
        for target in available_targets:
            target_info.append({
                "id": target["id"],
                "tags": target.get("tags", ""),
                "description": f"Target #{target['id']} - {target.get('tags', 'No tags')}"
            })
        
        prompt = f"""
You are an expert in Associative Remote Viewing target selection. 
Select the most optimal targets for this trial:

Domain: {domain}
Outcomes: {json.dumps(outcomes)}
Available Targets: {json.dumps(target_info[:20])}  # Limit to first 20 for prompt size

Select targets that are:
1. Maximally distinctive from each other
2. Clear and unambiguous
3. Appropriate for ARV protocols
4. Well-suited to the trial domain

Provide response in JSON format:
{{
    "recommended_targets": [
        {{"outcome": "outcome_name", "target_id": target_id, "reasoning": "why this target works well"}}
    ],
    "selection_principles": ["principle1", "principle2", ...],
    "warnings": ["any potential issues with selected targets"]
}}
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "You are an expert in ARV target selection. Provide specific target recommendations in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        suggestions = json.loads(response.choices[0].message.content)
        return {"suggestions": suggestions, "success": True}
        
    except Exception as e:
        return {"error": f"Failed to generate target suggestions: {str(e)}", "success": False}

def suggest_timing_optimization(title: str, domain: str, proposed_time: str) -> Dict[str, Any]:
    """
    Analyze and suggest optimal timing for trial resolution.
    """
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key"}
    
    try:
        prompt = f"""
Analyze the timing for this ARV prediction trial:

Title: {title}
Domain: {domain}
Proposed Resolution Time: {proposed_time}

Consider factors like:
- Market hours (for financial trials)
- Sports scheduling patterns
- Lottery draw times
- Optimal prediction-to-outcome windows for ARV
- Time zone considerations

Provide suggestions in JSON format:
{{
    "timing_analysis": "analysis of the proposed time",
    "optimal_window": "recommended time range for resolution", 
    "considerations": ["factor1", "factor2", ...],
    "alternative_times": ["alternative1", "alternative2", ...] if applicable,
    "confidence_factors": "what makes timing predictions more reliable"
}}
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "You are an expert in ARV trial timing optimization. Provide practical timing advice in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        suggestions = json.loads(response.choices[0].message.content)
        return {"suggestions": suggestions, "success": True}
        
    except Exception as e:
        return {"error": f"Failed to generate timing suggestions: {str(e)}", "success": False}

def analyze_trial_viability(title: str, domain: str, description: str, outcomes: List[str]) -> Dict[str, Any]:
    """
    Analyze the overall viability and potential issues with a trial configuration.
    """
    if not AI_ENABLED:
        return {"error": "AI suggestions require OpenAI API key"}
    
    try:
        prompt = f"""
Analyze this ARV trial configuration for potential issues and improvements:

Title: {title}
Domain: {domain}
Description: {description}
Outcomes: {json.dumps(outcomes)}

Evaluate:
1. Clarity and specificity of the trial
2. Resolvability - can outcomes be clearly determined?
3. ARV suitability - appropriate for remote viewing protocols?
4. Potential ambiguities or problems
5. Overall trial quality score (1-10)

Provide analysis in JSON format:
{{
    "viability_score": score_1_to_10,
    "strengths": ["strength1", "strength2", ...],
    "potential_issues": ["issue1", "issue2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "resolvability_analysis": "how clearly can this trial be resolved?",
    "arv_suitability": "how well does this work for remote viewing?"
}}
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "You are an expert ARV trial analyst. Provide thorough viability analysis in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        suggestions = json.loads(response.choices[0].message.content)
        return {"suggestions": suggestions, "success": True}
        
    except Exception as e:
        return {"error": f"Failed to analyze trial viability: {str(e)}", "success": False}