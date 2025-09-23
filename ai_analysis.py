import os
import json
import base64
# OpenAI import handled in initialization block below
from typing import Dict, List, Optional, Any
import sqlite3
import psycopg2
from dataclasses import dataclass

# Database connection helper
def get_db_connection():
    """Get PostgreSQL database connection using environment DATABASE_URL"""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)

# Initialize OpenAI client with basic configuration
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                timeout=30.0
            )
            print("OpenAI client initialized successfully")
        except ImportError:
            print("OpenAI package not available - fallback analysis will be used")
            client = None
    else:
        client = None
except Exception as e:
    print(f"OpenAI client initialization failed: {e}")
    client = None

@dataclass
class MatchingResult:
    target_id: int
    target_name: str
    overall_match: float
    category_matches: Dict[str, float]
    reasoning: str

@dataclass
class AIAnalysisResult:
    results: List[MatchingResult]
    recommended_target: MatchingResult
    analysis_summary: str

class AIAnalysisEngine:
    def __init__(self):
        # Use lowercase categories to match frontend form fields
        self.descriptor_categories = [
            "colours", "tactile", "energy", "smell", "sound", "visual"
        ]
    
    def analyze_viewing_session(self, trial_id: int, user_id: int, outcome_id: int = None, saved_descriptor_ids: List[int] = None) -> Optional[AIAnalysisResult]:
        """Analyze a user's viewing session against all possible targets"""
        try:
            # Get trial and target information
            trial_data = self._get_trial_data(trial_id)
            if not trial_data:
                return None
            
            # Get user's descriptors and drawings
            user_data = self._get_user_viewing_data(trial_id, user_id, outcome_id, saved_descriptor_ids)
            if not user_data:
                return None
            
            # Get all possible targets for this trial
            targets = self._get_trial_targets(trial_id)
            if not targets:
                return None
            
            # Analyze each target
            results = []
            for target in targets:
                match_result = self._analyze_target_match(user_data, target, trial_data)
                if match_result:
                    results.append(match_result)
            
            if not results:
                return None
            
            # Find the best match
            best_match = max(results, key=lambda x: x.overall_match)
            
            # Generate analysis summary
            summary = self._generate_analysis_summary(results, best_match)
            
            return AIAnalysisResult(
                results=results,
                recommended_target=best_match,
                analysis_summary=summary
            )
            
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return None
    
    def _get_trial_data(self, trial_id: int) -> Optional[Dict]:
        """Get trial information from database"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, domain
                FROM trial WHERE id = %s
            """, (trial_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'title': result[0],
                    'domain': result[1], 
                    'outcome_names': {}  # Will get from outcomes table
                }
            return None
            
        except Exception as e:
            print(f"Error getting trial data: {e}")
            return None
    
    def _fallback_target_match(self, user_data: Dict, target: Dict) -> Optional[MatchingResult]:
        """Simple fallback analysis when OpenAI is unavailable"""
        try:
            import re
            target_tags = re.split(r'[,;\s]+', target.get('tags', '').lower())
            target_tags = [tag.strip() for tag in target_tags if tag.strip()]
            if not target_tags:
                target_tags = []
            
            category_matches = {}
            total_score = 0.0
            category_count = 0
            
            # Simple keyword matching per category
            for category in self.descriptor_categories:
                if category in user_data.get('descriptors', {}):
                    descriptors = user_data['descriptors'][category]
                    category_score = 0.0
                    
                    for desc in descriptors:
                        content_words = desc['content'].lower().split()
                        # Count overlapping words
                        matches = sum(1 for word in content_words if word in target_tags)
                        if content_words:
                            category_score = max(category_score, matches / len(content_words))
                    
                    category_matches[category] = round(category_score, 3)
                    total_score += category_score
                    category_count += 1
                else:
                    category_matches[category] = 0.0
            
            overall_match = round(total_score / max(category_count, 1), 3)
            
            return MatchingResult(
                target_id=target['id'],
                target_name=target.get('filename', 'Unknown'),
                overall_match=overall_match,
                category_matches=category_matches,
                reasoning=f"Simple keyword matching analysis (OpenAI unavailable). Found {overall_match}% similarity based on word overlap with target tags."
            )
            
        except Exception as e:
            print(f"Fallback analysis failed: {e}")
            return None

    def _normalize_category(self, category: str) -> str:
        """Normalize category names to standard lowercase British spelling"""
        category = category.lower().strip()
        # Map common variants to standard names
        category_map = {
            'colors': 'colours',
            'color': 'colours',
            'colours': 'colours',
            'tactile': 'tactile',
            'energy': 'energy',
            'smell': 'smell',
            'sound': 'sound',
            'visual': 'visual'
        }
        return category_map.get(category, category)

    def _get_user_viewing_data(self, trial_id: int, user_id: int, outcome_id: int = None, saved_descriptor_ids: List[int] = None) -> Optional[Dict]:
        """Get user's descriptors and drawings for the trial"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # First get the user's name for filtering (since author field stores name, not ID)
            cursor.execute('SELECT name FROM "user" WHERE id = %s', (user_id,))
            user_result = cursor.fetchone()
            if not user_result:
                print(f"User not found: {user_id}")
                return None
            user_name = user_result[0]
            
            # If we have specific descriptor IDs from the just-saved request, use those
            if saved_descriptor_ids:
                id_placeholders = ','.join(['%s' for _ in saved_descriptor_ids])
                cursor.execute(f"""
                    SELECT category, text, created_at
                    FROM consensusdescriptor 
                    WHERE id IN ({id_placeholders})
                    ORDER BY created_at DESC
                """, saved_descriptor_ids)
                print(f"Using specific descriptor IDs: {saved_descriptor_ids}")
            elif outcome_id:
                # Filter by specific outcome and user
                cursor.execute("""
                    SELECT category, text, created_at
                    FROM consensusdescriptor 
                    WHERE trial_id = %s AND outcome_id = %s AND author = %s
                    AND created_at >= NOW() - INTERVAL '5 minutes'
                    ORDER BY created_at DESC
                """, (trial_id, outcome_id, user_name))
                print(f"Using outcome filter: trial={trial_id}, outcome={outcome_id}, user={user_name}")
            else:
                # Fallback to recent descriptors by user (less reliable)
                cursor.execute("""
                    SELECT category, text, created_at
                    FROM consensusdescriptor 
                    WHERE trial_id = %s AND author = %s
                    AND created_at >= NOW() - INTERVAL '5 minutes'
                    ORDER BY created_at DESC
                """, (trial_id, user_name))
                print(f"Using fallback filter: trial={trial_id}, user={user_name}")
            
            descriptors = {}
            for row in cursor.fetchall():
                category, content, created_at = row
                normalized_category = self._normalize_category(category)
                if normalized_category not in descriptors:
                    descriptors[normalized_category] = []
                descriptors[normalized_category].append({
                    'content': content,
                    'created_at': created_at
                })
            
            # Get drawings (simplified - drawings table not implemented yet)
            drawings = []
            
            conn.close()
            
            print(f"Retrieved {sum(len(cat_list) for cat_list in descriptors.values())} descriptors for user {user_name} in trial {trial_id}")
            
            return {
                'descriptors': descriptors,
                'drawings': drawings
            }
            
        except Exception as e:
            print(f"Error getting user viewing data: {e}")
            return None
    
    def _get_trial_targets(self, trial_id: int) -> Optional[List[Dict]]:
        """Get all targets for the trial"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT t.id, t.uri, t.tags, tro.label
                FROM target t
                JOIN trialtarget tt ON t.id = tt.target_id
                JOIN trialoutcome tro ON tt.outcome_id = tro.id
                WHERE tro.trial_id = %s
            """, (trial_id,))
            
            targets = []
            for row in cursor.fetchall():
                target_id, uri, tags, outcome_label = row
                
                # Read image file
                image_path = f"static/{uri}"
                image_data = None
                if os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode()
                
                targets.append({
                    'id': target_id,
                    'filename': uri,
                    'tags': tags if tags else "",
                    'outcome_name': outcome_label,
                    'image_data': image_data
                })
            
            conn.close()
            return targets
            
        except Exception as e:
            print(f"Error getting trial targets: {e}")
            return None
    
    def _analyze_target_match(self, user_data: Dict, target: Dict, trial_data: Dict) -> Optional[MatchingResult]:
        """Analyze how well user's viewing data matches a specific target"""
        try:
            if not client:
                print("OpenAI client not available - using fallback analysis")
                return self._fallback_target_match(user_data, target)
                
            # Prepare the analysis prompt
            prompt = self._build_analysis_prompt(user_data, target, trial_data)
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in Associative Remote Viewing (ARV) analysis. Analyze how well the viewer's descriptions match the target image across different sensory categories. Provide detailed percentage matches and reasoning."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{target['image_data']}"
                                }
                            } if target['image_data'] else {"type": "text", "text": "[No image available]"}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            content = response.choices[0].message.content
            if not content:
                print("Empty response from OpenAI")
                return None
            result = json.loads(content)
            
            return MatchingResult(
                target_id=target['id'],
                target_name=target['outcome_name'],
                overall_match=float(result.get('overall_match', 0)),
                category_matches={
                    category: float(result.get('category_matches', {}).get(category, 0))
                    for category in self.descriptor_categories
                },
                reasoning=result.get('reasoning', '')
            )
            
        except Exception as e:
            print(f"Error analyzing target match: {e}")
            return None
    
    def _build_analysis_prompt(self, user_data: Dict, target: Dict, trial_data: Dict) -> str:
        """Build the analysis prompt for OpenAI"""
        prompt = f"""
Analyze how well this remote viewing session matches the target image.

TRIAL CONTEXT:
- Title: {trial_data['title']}
- Domain: {trial_data['domain']}
- Target: {target['outcome_name']}

VIEWER'S DESCRIPTORS:
"""
        
        # Add descriptors by category
        for category in self.descriptor_categories:
            if category in user_data['descriptors']:
                prompt += f"\n{category.upper()}:\n"
                for desc in user_data['descriptors'][category]:
                    # Note: ball_number not available for regular ARV sessions, only lottery trials
                    prompt += f"- {desc['content']}\n"
            else:
                prompt += f"\n{category.upper()}: No descriptions provided\n"
        
        # Add drawings if available
        if user_data['drawings']:
            prompt += f"\nDRAWINGS: {len(user_data['drawings'])} drawing(s) provided\n"
            for i, drawing in enumerate(user_data['drawings'], 1):
                if drawing['description']:
                    prompt += f"Drawing {i} description: {drawing['description']}\n"
        
        # Add target tags for reference
        if target['tags']:
            prompt += f"\nTARGET TAGS: {', '.join(target['tags'])}\n"
        
        prompt += """
ANALYSIS REQUIRED:
Provide a JSON response with:
1. "overall_match": Overall percentage match (0-100)
2. "category_matches": Object with percentage matches for each category:
   - "Colors": 0-100
   - "Tactile": 0-100  
   - "Energy": 0-100
   - "Smell": 0-100
   - "Sound": 0-100
   - "Visual": 0-100
3. "reasoning": Detailed explanation of the analysis

Consider:
- How well the descriptors match what you see in the image
- Accuracy of colors, shapes, textures described
- Energy and emotional qualities
- Any sounds or smells that would be associated with the image
- Overall visual correspondence

Be objective and detailed in your analysis.
"""
        
        return prompt
    
    def _generate_analysis_summary(self, results: List[MatchingResult], best_match: MatchingResult) -> str:
        """Generate a summary of the analysis"""
        total_targets = len(results)
        avg_match = sum(r.overall_match for r in results) / total_targets if results else 0
        
        summary = f"""
ANALYSIS SUMMARY:
- Analyzed {total_targets} possible targets
- Average match rate: {avg_match:.1f}%
- Recommended target: {best_match.target_name} ({best_match.overall_match:.1f}% match)

The analysis shows strongest correspondence with {best_match.target_name}, particularly in:
"""
        
        # Find strongest category matches
        strong_categories = [
            cat for cat, score in best_match.category_matches.items() 
            if score >= 70
        ]
        
        if strong_categories:
            summary += ", ".join(strong_categories) + " categories."
        else:
            summary += "general visual elements."
        
        return summary

def save_analysis_result(trial_id: int, user_id: int, analysis: AIAnalysisResult) -> bool:
    """Save analysis result to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create analysis results table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER,
                user_id INTEGER,
                analysis_data TEXT,
                recommended_target_id INTEGER,
                overall_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trial_id) REFERENCES trials (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Save the analysis
        analysis_json = json.dumps({
            'results': [
                {
                    'target_id': r.target_id,
                    'target_name': r.target_name,
                    'overall_match': r.overall_match,
                    'category_matches': r.category_matches,
                    'reasoning': r.reasoning
                }
                for r in analysis.results
            ],
            'analysis_summary': analysis.analysis_summary
        })
        
        cursor.execute("""
            INSERT INTO analysis_results 
            (trial_id, user_id, analysis_data, recommended_target_id)
            VALUES (?, ?, ?, ?)
        """, (
            trial_id,
            user_id, 
            analysis_json,
            analysis.recommended_target.target_id
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error saving analysis result: {e}")
        return False

def get_analysis_result(trial_id: int, user_id: int) -> Optional[AIAnalysisResult]:
    """Get saved analysis result from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT analysis_data, recommended_target_id
            FROM analysis_results
            WHERE trial_id = %s AND user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (trial_id, user_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            analysis_data = json.loads(result[0])
            
            # Reconstruct MatchingResult objects
            results = []
            for r_data in analysis_data['results']:
                results.append(MatchingResult(
                    target_id=r_data['target_id'],
                    target_name=r_data['target_name'],
                    overall_match=r_data['overall_match'],
                    category_matches=r_data['category_matches'],
                    reasoning=r_data['reasoning']
                ))
            
            # Find recommended target
            recommended_target = next(
                (r for r in results if r.target_id == result[1]), 
                results[0] if results else None
            )
            
            if not recommended_target:
                return None
            
            return AIAnalysisResult(
                results=results,
                recommended_target=recommended_target,
                analysis_summary=analysis_data['analysis_summary']
            )
        
        return None
        
    except Exception as e:
        print(f"Error getting analysis result: {e}")
        return None