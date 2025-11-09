#!/usr/bin/env python3.12
"""
Car Identifier - PySide6 (Qt) experiment - Python 3.2+ Optimized
Lightweight, enhanced-only streaming UI with threadpool workers and a scrollable results list.

Dependencies: PySide6, pillow, ollama
Run: python car_identifier_gui_qt_v3_final_merged.py
"""

import sys
import os
import io
import json
import base64
import time
import threading
import re
from pathlib import Path

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Transparent Ollama performance defaults (GPU offload, flash attention, keep-alive)
try:
    os.environ.setdefault('OLLAMA_HOST', 'http://127.0.0.1:11434')
    os.environ.setdefault('OLLAMA_GPU_LAYERS', '-1')          # max GPU offload
    os.environ.setdefault('OLLAMA_FLASH_ATTENTION', 'true')   # if supported by GPU
    os.environ.setdefault('OLLAMA_KEEP_ALIVE', '30m')         # keep model hot between calls
    os.environ.setdefault('OLLAMA_NUM_THREADS', '8')          # optional CPU thread cap
except Exception:
    pass

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QSettings

import ollama
import requests
from secret_cloud import cloud_router

# Global logging function
def _log(msg: str) -> None:
    """Log message to log.log file with timestamp"""
    try:
        with open(os.path.join(os.getcwd(), 'log.log'), 'a', encoding='utf-8') as lf:
            lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] {msg}\n")
    except Exception:
        pass

DEBUG_META: bool = False
APP_VERSION: str = "v4.2"
# Unified summary key for EXIF JSON and UI
SUMMARY_KEY: str = "TagoMatic AI Summary"
VERSION_CHECK_URL: str = "https://www.tagomatic.co.uk/version.json"

# Prompt Template System
PROMPT_TEMPLATES = {
    'vehicle_classification': {
        'local': 'Analyze the primary subject in the provided image. Respond with EXACTLY one word: Car, Motorbike, or Neither. If uncertain, respond with "Neither".',
        'cloud': 'Return ONLY: Car, Motorbike, or Neither. No other text.',
        'fallback': 'What type of vehicle is this? Answer in one word.'
    },
    'scene_analysis': {
        'local': 'Analyze this image and return ONLY these 5 fields in exact order:\nImage Type: <Photograph|Digital Art|Illustration|Drawing|Other>\nPrimary Subject: <Main person/animal/object visible>\nSetting: <Environment/background>\nKey Objects: <Comma-separated list of other visible items>\nDescriptive Summary: <Condensed version, 150 chars max>\n\nRules: Be specific but concise. Base answers on what is actually visible.',
        'cloud': 'Return ONLY these 5 fields in exact order:\nImage Type: <Photograph|Digital Art|Illustration|Drawing|Other>\nPrimary Subject: <Main person/animal/object visible>\nSetting: <Environment/background>\nKey Objects: <Comma-separated list of other visible items>\nDescriptive Summary: <Condensed version, 150 chars max>',
        'fallback': 'Describe this image in 5 structured fields: Type, Subject, Setting, Objects, Summary.'
    },
    'car_analysis': {
        'local': 'Identify this car and provide these exact details:\n\nAnalyze the car systematically by examining these features in order:\n1. Check badge/model text visible on the car (most reliable - extract ALL text)\n2. Examine grille pattern (honeycomb, mesh, horizontal bars, etc.)\n3. Examine headlight design (shape, integration, position)\n4. Examine taillight design (shape, position, stacking)\n5. Examine side vents/intakes (location, shape)\n6. Examine body shape (fastback, coupe, angular, smooth)\n7. Examine exhaust configuration (dual, quad, central, side)\n8. Look for other unique design elements (doors, spoilers, diffusers)\n\nCommon exotic/rare models to recognize:\n- Pagani: Zonda (exposed quad exhaust, distinctive side mirrors), Huayra (smooth body, active aero)\n- Koenigsegg: CCX, Agera, Regera (distinctive wraparound windscreen)\n- TVR: Identify by examining features systematically - Grille: Honeycomb = Cerberus/Cerbera; Headlights: Oval integrated = Cerberus/Cerbera; Taillights: Vertically stacked = Cerberus/Cerbera; Side vents: On doors = Cerberus/Cerbera, Near wheel arches = Sagaris; Body: Long fastback = Cerberus/Cerbera, Aggressive angular = Sagaris, Smooth = Tuscan\n- Noble: M12, M600 (British supercar, angular design)\n\nFor Ferrari models:\n- FF/GTC4Lusso: shooting brake body (hatchback)\n- 430/458/488: traditional mid-engine coupes\n- Enzo: gullwing doors, extreme wedge\n\nMake: (brand name)\nModel: (specific model name - be precise)\nColor: (main body color)\nLogos: (comma-separated list of ALL visible logos, sponsors, brands, text on car - check hood, doors, roof, wing, wheels, windshield. Be thorough!)\nRace Number: (check doors, roof, hood, wing, windows for numbers - look carefully)\nLicense Plate: (check front AND rear plates)\nAI-Interpretation Summary: (descriptive narrative, up to 500 characters)\n\nPlease provide each field on its own line exactly as shown above. Be thorough with Logos - list EVERY visible brand/sponsor/text.',
        'cloud': 'Identify this car from the image. Analyze the vehicle systematically:\n1. Check for visible badges, logos, or model text on the car\n2. Examine body style, distinctive features, and design elements\n3. Look for race numbers on doors, roof, or hood\n4. Check for license plates (front and rear)\n\nFor Porsche: Look for model badges (911, GT3, GT3 RS, Turbo, etc.) and check for distinctive features like large rear wings.\n\nFor Ferrari: Badge "430"/"F430"/"Scuderia" = 430 Scuderia (NOT Enzo). Badge "458" = 458 Italia. Badge "Enzo" or gullwing doors = Enzo.\n\nYou MUST return exactly these 7 lines in this exact format:\nMake: <brand name or Unknown>\nModel: <specific model name or Unknown>\nColor: <main body color or Unknown>\nLogos: <all visible badges, logos, sponsors, text on car or Unknown>\nRace Number: <number visible on car or Unknown>\nLicense Plate: <plate visible or Unknown>\nAI-Interpretation Summary: <detailed description of the car, its features, setting, and notable details - up to 500 characters>\n\nProvide complete information for each field. Do not skip any fields.',
        'motorshow': 'If unclear, prefer Unknown for fields; keep summary concise.',
        'racing': '''RACE MODE - This is a motorsport/racing scenario. Priority adjustments:

CRITICAL: Race Number extraction is the HIGHEST PRIORITY.
- Look for race numbers on: doors, roof, hood, windshield, side windows, rear window
- Race numbers are often large, brightly colored, and prominently displayed
- Check ALL visible surfaces - numbers may be on multiple locations
- Extract the number exactly as displayed (e.g., "79", "117", "555")
- If multiple numbers visible, identify the PRIMARY race number on the main vehicle

IMPORTANT: Sponsor/Logo extraction is HIGH PRIORITY for race cars.
- Extract ALL sponsor logos, brand names, and text visible on the car
- Check: hood, doors, roof, wing, side panels, front/rear bumpers
- Common race sponsors: MRF, Michelin, Pirelli, Castrol, Mobil, etc.
- Include team names, driver names if visible on car

Make/Model identification:
- Focus on distinctive racing modifications (wings, splitters, liveries)
- Race cars may have body modifications that differ from street versions
- If model unclear, note racing-specific features in summary

Summary should include:
- Race context (track, wet/dry conditions if visible)
- Racing modifications visible
- Sponsor liveries and branding
- Any visible race numbers or timing equipment'''
    },
    'motorbike_analysis': {
        'local': '''Act as an expert motorcycle analyst. Analyze this image and identify motorbike details:

Make: [brand]
Model: [model] 
Type: [Sport/Street/Cruiser/Touring/Off-road/Other]
Engine Size: [cc if visible/estimatable]
Color: [color]
Logos: [comma-separated visible badges/emblems and any readable brand/sponsor text; quote text EXACTLY]
Race Number: [number if visible - CHECK ALL WINDOWS (windshield, side windows, rear window) AND CENTER OF DOOR PANELS for bright colored stickers/numbers]
License Plate: [plate if visible]
AI-Interpretation Summary: [up to 500 chars]

Focus on distinctive features: exhaust position, fairing design, wheel configuration, handlebar type, seat design.''',
        'cloud': 'Return ONLY these 9 lines in exact order:\nMake: <brand>\nModel: <model>\nType: <type>\nEngine Size: <cc>\nColor: <color>\nLogos: <logos>\nRace Number: <number - CHECK ALL WINDOWS AND CENTER OF DOOR PANELS for bright stickers/numbers>\nLicense Plate: <plate>\nAI-Interpretation Summary: <summary>',
        'fallback': 'Describe this motorcycle in detail, focusing on visible features and characteristics.',
        'racing': 'In racing scenes, focus on racing-specific features: fairings, numbers, sponsor liveries, and racing modifications. Engine size may be visible on fairings or number plates.',
        'off_road': 'In off-road scenes, focus on terrain-specific features: knobby tires, high ground clearance, protective guards, and off-road modifications.',
        'urban': 'In urban scenes, focus on street-legal features: mirrors, lights, license plates, and street-appropriate modifications.'
    },
    'non_car_analysis': {
        'local': 'Analyze this non-vehicle image and provide a rich, descriptive scene analysis (150-220 words). Focus on:\n- Main subjects and their characteristics\n- Environmental details and atmosphere\n- Composition and visual elements\n- Mood and ambiance\n\nBe descriptive but factual. Avoid speculation beyond what is visible.',
        'cloud': 'Provide a descriptive scene analysis (150-220 words) focusing on main subjects, environment, composition, and mood. Be factual and descriptive.',
        'fallback': 'Describe this scene in detail, focusing on what you can clearly see.'
    }
}

def get_prompt(template_name: str, context: dict = None) -> str:
    """Get appropriate prompt based on model type and context"""
    if template_name not in PROMPT_TEMPLATES:
        return "Please analyze this image."
    
    # Determine base prompt type
    if context and context.get('use_cloud'):
        base_prompt = PROMPT_TEMPLATES[template_name]['cloud']
    else:
        base_prompt = PROMPT_TEMPLATES[template_name]['local']
    
    # Add context-specific modifications
    if context and context.get('motorshow_mode') and 'motorshow' in PROMPT_TEMPLATES[template_name]:
        base_prompt += "\n\n" + PROMPT_TEMPLATES[template_name]['motorshow']
    
    # Add race mode modifications (takes priority over motorshow for race scenarios)
    if context and context.get('race_mode') and 'racing' in PROMPT_TEMPLATES[template_name]:
        base_prompt += "\n\n" + PROMPT_TEMPLATES[template_name]['racing']
    
    # Add motorbike-specific context modifications
    if template_name == 'motorbike_analysis' and context:
        if context.get('racing_scene') and 'racing' in PROMPT_TEMPLATES[template_name]:
            base_prompt += "\n\n" + PROMPT_TEMPLATES[template_name]['racing']
        if context.get('off_road') and 'off_road' in PROMPT_TEMPLATES[template_name]:
            base_prompt += "\n\n" + PROMPT_TEMPLATES[template_name]['off_road']
        if context.get('urban_scene') and 'urban' in PROMPT_TEMPLATES[template_name]:
            base_prompt += "\n\n" + PROMPT_TEMPLATES[template_name]['urban']
    
    # Add feedback context if available
    if context and context.get('feedback'):
        feedback = context['feedback']
        if isinstance(feedback, dict):
            reason = feedback.get('reason', '').strip()
            comments = feedback.get('comments', '').strip()
            correct_make = feedback.get('correct_make', '').strip()
            correct_model = feedback.get('correct_model', '').strip()
            distinguishing_cues = feedback.get('distinguishing_cues', {})
            
            # If user provided correct make/model, create STRONG correction directive
            if correct_make and correct_model:
                base_prompt += f"\n\n{'='*80}\n"
                base_prompt += f"⚠️ CRITICAL CORRECTION - USER VERIFIED IDENTIFICATION ⚠️\n"
                base_prompt += f"{'='*80}\n"
                base_prompt += f"THE PREVIOUS IDENTIFICATION WAS WRONG. The user has CONFIRMED that this vehicle is:\n"
                base_prompt += f"** MAKE: {correct_make.upper()} **\n"
                base_prompt += f"** MODEL: {correct_model.upper()} **\n"
                base_prompt += f"\nYOU MUST identify this vehicle as {correct_make} {correct_model}.\n"
                base_prompt += f"Do NOT identify it as anything else.\n"
                
                # Add distinguishing cues if available
                if distinguishing_cues and isinstance(distinguishing_cues, dict):
                    base_prompt += f"\nDistinguishing features to look for:\n"
                    for angle, desc in distinguishing_cues.items():
                        if desc and desc.strip():
                            base_prompt += f"- {angle.capitalize()}: {desc}\n"
                
                base_prompt += f"\nRejection reason: {reason or 'Incorrect identification'}\n"
                if comments:
                    base_prompt += f"User comments: {comments}\n"
                base_prompt += f"{'='*80}\n"
            elif reason or comments:
                # Standard feedback without correct make/model
                base_prompt += f"\n\nUser feedback (use to correct previous mistake; prioritize what is actually visible):\n- Rejection reason: {reason or 'n/a'}\n- Comments: {comments or 'n/a'}"
    
    # Add KB hints if available (proactive identification hints from user KB)
    if context and context.get('kb_hints'):
        kb_hints = context['kb_hints']
        if kb_hints and kb_hints.strip():
            base_prompt += kb_hints
    
    return base_prompt

def build_contextual_prompt(base_prompt: str, context: dict) -> str:
    """Build context-aware prompts based on image characteristics"""
    
    # Detect image characteristics
    if context.get('low_light'):
        base_prompt += "\nNote: This image appears to be in low light conditions. Focus on visible details."
    
    if context.get('crowded_scene'):
        base_prompt += "\nThis appears to be a crowded scene. Prioritize the main subject over background details."
    
    if context.get('partial_vehicle'):
        base_prompt += "\nThe vehicle appears to be partially visible. Only identify what you can clearly see."
    
    # Model-specific adjustments
    if context.get('model_type') == 'vision_only':
        base_prompt += "\nFocus on visual analysis only. Avoid speculation about non-visible details."
    
    return base_prompt

def smart_retry_with_fallback(client, model_name: str, prompt: str, img_b64: str, 
                             max_retries: int = 2) -> str:
    """Fast single-strategy approach for better performance"""
    
    # Single fast strategy instead of multiple retries
    strategy = {'temperature': 0.2, 'top_p': 0.8, 'num_predict': 64}  # Fast and efficient
    
    try:
        response = chat(client, model_name, 
                      [{'role': 'user', 'content': prompt, 'images': [img_b64]}], 
                      options_override=strategy)
        result = extract_message_text(response)
        if result and len(result.strip()) > 5:  # Basic validation
            return result
        else:
            return "Analysis failed - insufficient response"
    except Exception as e:
        print(f"Smart retry failed: {e}")
        return "Analysis failed - unable to process image"

def unified_scene_analysis(client, model_name: str, img_b64: str, context: dict = None) -> dict:
    """Single LLM call with structured output - more efficient than 2-phase approach"""
    
    # Get appropriate prompt template
    prompt = get_prompt('scene_analysis', context)
    
    # Add contextual modifications
    if context:
        prompt = build_contextual_prompt(prompt, context)
    
    try:
        # Single call with smart retry
        response = smart_retry_with_fallback(client, model_name, prompt, img_b64)
        return parse_structured_scene_response(response)
    except Exception as e:
        print(f"Unified scene analysis failed: {e}")
        return fallback_scene_analysis()

def parse_structured_scene_response(response: str) -> dict:
    """Parse the structured response from unified scene analysis"""
    
    default_output = {
        'Image Type': 'Photograph',
        'Primary Subject': '',
        'Setting': '',
        'Key Objects': '',
        'Descriptive Summary': ''
    }
    
    if not response:
        return default_output
    
    lines = [ln.strip() for ln in response.split('\n') if ln.strip()]
    
    for ln in lines:
        if ':' not in ln:
            continue
        k, v = ln.split(':', 1)
        key = k.strip().lower()
        val = v.strip().strip('"')
        
        if key.startswith('image type'):
            default_output['Image Type'] = val or 'Photograph'
        elif key.startswith('primary subject'):
            default_output['Primary Subject'] = val or ''
        elif key.startswith('setting'):
            default_output['Setting'] = val or ''
        elif key.startswith('key objects'):
            default_output['Key Objects'] = val or ''
        elif key.startswith('descriptive summary'):
            default_output['Descriptive Summary'] = (val or '')[:200]
    
    return default_output

def fallback_scene_analysis() -> dict:
    """Fallback response when scene analysis fails"""
    return {
        'Image Type': 'Photograph',
        'Primary Subject': 'Image Content',
        'Setting': 'Image Environment', 
        'Key Objects': '',
        'Descriptive Summary': 'Image analysis failed - invalid data'
    }

def validate_and_correct_response(response: str, expected_format: str) -> str:
    """Validate LLM response and attempt self-correction"""
    
    if expected_format == 'vehicle_count':
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            count = int(numbers[0])
            return str(max(0, min(5, count)))  # Clamp to 0-5
    
    elif expected_format == 'structured_fields':
        # Check if response has required field markers
        required_fields = ['Image Type:', 'Primary Subject:', 'Setting:']
        if not all(field in response for field in required_fields):
            # Attempt to reformat
            return reformat_structured_response(response)
    
    return response

def reformat_structured_response(response: str) -> str:
    """Attempt to reformat malformed structured responses"""
    
    # Simple reformatting logic
    if 'image' in response.lower() and 'subject' in response.lower():
        # Try to extract and reformat
        lines = []
        if 'image' in response.lower():
            lines.append('Image Type: Photograph')
        if 'subject' in response.lower():
            lines.append('Primary Subject: Image Content')
        if 'setting' in response.lower() or 'background' in response.lower():
            lines.append('Setting: Image Environment')
        if 'objects' in response.lower() or 'items' in response.lower():
            lines.append('Key Objects: ')
        if 'summary' in response.lower() or 'description' in response.lower():
            lines.append('Descriptive Summary: Image analysis completed')
        
        return '\n'.join(lines)
    
    return response

class PromptMetrics:
    """Track prompt effectiveness and provide optimization suggestions"""
    
    def __init__(self):
        self.success_rates = {}
        self.avg_response_times = {}
        self.fallback_usage = {}
        self.total_attempts = {}
    
    def record_prompt_result(self, prompt_type: str, success: bool, response_time: float):
        """Record the result of a prompt attempt"""
        if prompt_type not in self.success_rates:
            self.success_rates[prompt_type] = []
            self.avg_response_times[prompt_type] = []
            self.total_attempts[prompt_type] = 0
        
        self.success_rates[prompt_type].append(success)
        self.avg_response_times[prompt_type].append(response_time)
        self.total_attempts[prompt_type] += 1
    
    def record_fallback_usage(self, prompt_type: str):
        """Record when fallback responses are used"""
        if prompt_type not in self.fallback_usage:
            self.fallback_usage[prompt_type] = 0
        self.fallback_usage[prompt_type] += 1
    
    def get_success_rate(self, prompt_type: str) -> float:
        """Get success rate for a specific prompt type"""
        if prompt_type not in self.success_rates or not self.success_rates[prompt_type]:
            return 0.0
        return sum(self.success_rates[prompt_type]) / len(self.success_rates[prompt_type])
    
    def get_avg_response_time(self, prompt_type: str) -> float:
        """Get average response time for a specific prompt type"""
        if prompt_type not in self.avg_response_times or not self.avg_response_times[prompt_type]:
            return 0.0
        return sum(self.avg_response_times[prompt_type]) / len(self.avg_response_times[prompt_type])
    
    def get_optimization_suggestions(self) -> list:
        """Analyze metrics and suggest prompt improvements"""
        suggestions = []
        
        for prompt_type in self.success_rates:
            success_rate = self.get_success_rate(prompt_type)
            avg_time = self.get_avg_response_time(prompt_type)
            fallback_count = self.fallback_usage.get(prompt_type, 0)
            total_attempts = self.total_attempts.get(prompt_type, 0)
            
            if success_rate < 0.8:
                suggestions.append(f"Prompt '{prompt_type}' has low success rate ({success_rate:.1%}) - consider simplifying or clarifying")
            
            if avg_time > 5.0:  # More than 5 seconds
                suggestions.append(f"Prompt '{prompt_type}' is slow ({avg_time:.1f}s avg) - consider reducing complexity")
            
            if fallback_count > 0 and total_attempts > 10:
                fallback_rate = fallback_count / total_attempts
                if fallback_rate > 0.2:  # More than 20% fallback usage
                    suggestions.append(f"Prompt '{prompt_type}' has high fallback rate ({fallback_rate:.1%}) - may need better error handling")
        
        return suggestions
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics for all prompt types"""
        stats = {}
        for prompt_type in self.success_rates:
            stats[prompt_type] = {
                'success_rate': self.get_success_rate(prompt_type),
                'avg_response_time': self.get_avg_response_time(prompt_type),
                'fallback_count': self.fallback_usage.get(prompt_type, 0),
                'total_attempts': self.total_attempts.get(prompt_type, 0)
            }
        return stats
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.success_rates.clear()
        self.avg_response_times.clear()
        self.fallback_usage.clear()
        self.total_attempts.clear()

# Global metrics instance
prompt_metrics = PromptMetrics()

# Enhanced Motorbike Detection Functions
def detect_motorbike_context(image_analysis: dict) -> dict:
    """Detect motorbike-specific context from image analysis"""
    
    context = {}
    
    # Check for motorbike-specific indicators
    scene_text = image_analysis.get('Setting', '').lower()
    objects_text = image_analysis.get('Key Objects', '').lower()
    
    # Racing indicators
    if any(word in scene_text for word in ['track', 'racing', 'circuit', 'motorsport', 'race']):
        context['racing_scene'] = True
    
    # Off-road indicators
    if any(word in scene_text for word in ['dirt', 'trail', 'off-road', 'mountain', 'terrain']):
        context['off_road'] = True
    
    # Urban indicators
    if any(word in scene_text for word in ['street', 'city', 'urban', 'traffic', 'road']):
        context['urban_scene'] = True
    
    # Check for motorbike-specific objects
    if any(word in objects_text for word in ['helmet', 'leathers', 'gloves', 'boots', 'gear']):
        context['rider_gear'] = True
    
    # Check for motorbike-specific features
    if any(word in objects_text for word in ['handlebars', 'exhaust', 'fairing', 'wheels']):
        context['motorbike_features'] = True
    
    return context

def build_motorbike_contextual_prompt(base_prompt: str, context: dict) -> str:
    """Build context-aware prompts for motorbikes"""
    
    # Add motorbike-specific context
    if context.get('racing_scene'):
        base_prompt += "\n\n" + PROMPT_TEMPLATES['motorbike_analysis']['racing']
    
    if context.get('off_road'):
        base_prompt += "\n\n" + PROMPT_TEMPLATES['motorbike_analysis']['off_road']
    
    if context.get('urban_scene'):
        base_prompt += "\n\n" + PROMPT_TEMPLATES['motorbike_analysis']['urban']
    
    # Add general context modifications
    if context.get('low_light'):
        base_prompt += "\n\nNote: Low light conditions. Focus on visible features like headlights, taillights, and reflective elements."
    
    if context.get('crowded_scene'):
        base_prompt += "\n\nNote: Multiple vehicles visible. Focus on the primary motorbike in the foreground."
    
    if context.get('partial_vehicle'):
        base_prompt += "\n\nNote: The motorbike appears to be partially visible. Only identify what you can clearly see."
    
    return base_prompt

def parse_motorbike_response(response: str) -> dict:
    """Parse motorbike analysis response with better error handling"""
    
    default_output = {
        'Make': 'Unknown',
        'Model': 'Unknown',
        'Type': 'Unknown',
        'Engine Size': 'Unknown',
        'Color': 'Unknown',
        'Logos': '',
        'Race Number': 'Unknown',
        'License Plate': 'Unknown',
        'AI-Interpretation Summary': ''
    }
    
    if not response:
        return default_output
    
    lines = [ln.strip() for ln in response.split('\n') if ln.strip()]
    
    for ln in lines:
        if ':' not in ln:
            continue
        k, v = ln.split(':', 1)
        key = k.strip().lower()
        val = v.strip().strip('"')
        
        # Map response fields to output
        if key.startswith('make'):
            default_output['Make'] = val or 'Unknown'
        elif key.startswith('model'):
            default_output['Model'] = val or 'Unknown'
        elif key.startswith('type'):
            default_output['Type'] = val or 'Unknown'
        elif key.startswith('engine'):
            default_output['Engine Size'] = val or 'Unknown'
        elif key.startswith('color'):
            default_output['Color'] = val or 'Unknown'
        elif key.startswith('logos'):
            default_output['Logos'] = val or ''
        elif key.startswith('race'):
            default_output['Race Number'] = val or 'Unknown'
        elif key.startswith('license'):
            default_output['License Plate'] = val or 'Unknown'
        elif key.startswith('ai-interpretation') or key.startswith('summary'):
            default_output['AI-Interpretation Summary'] = (val or '')[:500]
    
    return default_output

def fallback_motorbike_analysis() -> dict:
    """Fallback response when motorbike analysis fails"""
    return {
        'Make': 'Unknown',
        'Model': 'Unknown',
        'Type': 'Unknown',
        'Engine Size': 'Unknown',
        'Color': 'Unknown',
        'Logos': '',
        'Race Number': 'Unknown',
        'License Plate': 'Unknown',
        'AI-Interpretation Summary': 'Motorbike analysis failed - unable to process image'
    }

def analyze_motorbike_enhanced(client, model_name: str, images: list, context: dict = None) -> dict:
    """Enhanced motorbike analysis with specific features using three-crop system"""
    
    # Get motorbike-specific prompt
    prompt = get_prompt('motorbike_analysis', context)
    
    # Add contextual modifications for motorbikes
    if context:
        prompt = build_motorbike_contextual_prompt(prompt, context)
    
    try:
        # Use the same three-crop system as cars for better make/model identification
        # Note: We need to get the image path from context or use a different approach
        # For now, use the main image but enhance the prompt for better visual analysis
        
        # Enhanced motorbike identification prompt
        enhanced_prompt = prompt + """
        
        CRITICAL: Look for these specific visual identifiers:
        - Manufacturer stickers/badges (Honda, Suzuki, Yamaha, Kawasaki, BMW, Ducati, etc.)
        - Model name stickers (CBR, GSX, R1, Ninja, S1000RR, Panigale, etc.)
        - Distinctive fairing shapes and designs
        - Headlight and taillight designs
        - Color schemes and racing liveries
        - Any visible text or branding on the bike
        
        IMPORTANT: Do NOT look for license plates as motorbikes rarely have front plates.
        Focus on visual cues like stickers, badges, and distinctive features for identification.
        """
        
        # Use all images (main + three crops) for comprehensive motorbike identification
        response = chat(client, model_name, 
                       [{'role': 'user', 'content': enhanced_prompt, 'images': images}],
                       options_override={"temperature": 0.1, "top_p": 0.8, "num_predict": 128})
        return parse_motorbike_response(response)
    except Exception as e:
        print(f"Motorbike analysis failed: {e}")
        return fallback_motorbike_analysis()

def count_vehicles_by_type(client, model_name: str, img_b64: str, primary_type: str) -> tuple[int, int]:
    """Count vehicles by type with enhanced detection"""
    
    if primary_type == 'Motorbike':
        # Use motorbike-specific counting
        return count_motorbikes_enhanced(client, model_name, img_b64)
    elif primary_type == 'Car':
        # Use car-specific counting
        return count_cars_enhanced(client, model_name, img_b64)
    else:
        # Mixed or uncertain - use general counting
        return count_vehicles_general(client, model_name, img_b64)

def count_motorbikes_enhanced(client, model_name: str, img_b64: str) -> tuple[int, int]:
    """Enhanced motorbike counting with specific features"""
    
    prompt = """Count the vehicles in this image. Focus on:
- Cars: 4 wheels, enclosed cabin, steering wheel
- Motorbikes: 2 wheels, exposed rider position, handlebars

Return ONLY: "Cars: X, Motorbikes: Y" where X and Y are numbers."""
    
    try:
        response = smart_retry_with_fallback(client, model_name, prompt, img_b64)
        car_count, motorbike_count = parse_vehicle_count_response(response)
        return car_count, motorbike_count
    except:
        return 0, 1  # Fallback: assume at least one motorbike if primary type was motorbike

def count_cars_enhanced(client, model_name: str, img_b64: str) -> tuple[int, int]:
    """Enhanced car counting with specific features"""
    
    prompt = """Count the vehicles in this image. Focus on:
- Cars: 4 wheels, enclosed cabin, steering wheel
- Motorbikes: 2 wheels, exposed rider position, handlebars

Return ONLY: "Cars: X, Motorbikes: Y" where X and Y are numbers."""
    
    try:
        response = smart_retry_with_fallback(client, model_name, prompt, img_b64)
        car_count, motorbike_count = parse_vehicle_count_response(response)
        return car_count, motorbike_count
    except:
        return 1, 0  # Fallback: assume at least one car if primary type was car

def count_vehicles_general(client, model_name: str, img_b64: str) -> tuple[int, int]:
    """General vehicle counting when type is uncertain"""
    
    prompt = """Count the vehicles in this image. Focus on:
- Cars: 4 wheels, enclosed cabin, steering wheel
- Motorbikes: 2 wheels, exposed rider position, handlebars

Return ONLY: "Cars: X, Motorbikes: Y" where X and Y are numbers."""
    
    try:
        response = smart_retry_with_fallback(client, model_name, prompt, img_b64)
        car_count, motorbike_count = parse_vehicle_count_response(response)
        return car_count, motorbike_count
    except:
        return 0, 0  # Fallback: no vehicles detected

def parse_vehicle_count_response(response: str) -> tuple[int, int]:
    """Parse vehicle count response from LLM"""
    
    try:
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', response)
        
        if len(numbers) >= 2:
            car_count = int(numbers[0])
            motorbike_count = int(numbers[1])
        elif len(numbers) == 1:
            # If only one number, assume it's cars
            car_count = int(numbers[0])
            motorbike_count = 0
        else:
            car_count = 0
            motorbike_count = 0
        
        return car_count, motorbike_count
    except:
        return 0, 0

# Usage Examples for the New Prompt System:
# 
# 1. Get a prompt with context:
#    context = {'use_cloud': True, 'motorshow_mode': True}
#    prompt = get_prompt('car_analysis', context)
#
# 2. Use unified scene analysis:
#    result = unified_scene_analysis(client, model_name, img_b64, context)
#
# 3. Track performance:
#    start_time = time.time()
#    response = chat(client, model_name, [{'role': 'user', 'content': prompt, 'images': [img_b64]}])
#    response_time = time.time() - start_time
#    prompt_metrics.record_prompt_result('car_analysis', True, response_time)
#
# 4. Get optimization suggestions:
#    suggestions = prompt_metrics.get_optimization_suggestions()
#    print("Optimization suggestions:", suggestions)
#
# 5. Enhanced motorbike analysis:
#    motorbike_result = analyze_motorbike_enhanced(client, model_name, img_b64, context)
#
# 6. Vehicle counting by type:
#    car_count, motorbike_count = count_vehicles_by_type(client, model_name, img_b64, 'Motorbike')

def _resource_path(relative_name: str) -> str:
    """Resolve a bundled asset path (PyInstaller onefile/onedir) or fallback to relative/absolute input.

    Looks under an 'assets' directory next to the executable or inside the PyInstaller temp dir.
    """
    try:
        base = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
        cand = os.path.join(base, 'assets', relative_name)
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return relative_name
def _app_base_dir() -> str:
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _secrets_path() -> str:
    return os.path.join(_app_base_dir(), 'secrets.enc')


def _win_dpapi_protect(raw: bytes) -> bytes:
    try:
        if os.name != 'nt':
            return raw
        import ctypes
        from ctypes import wintypes
        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
        CryptProtectData = ctypes.windll.crypt32.CryptProtectData
        CryptProtectData.argtypes = [ctypes.POINTER(DATA_BLOB), wintypes.LPCWSTR, ctypes.POINTER(DATA_BLOB), wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
        CryptProtectData.restype = wintypes.BOOL
        in_blob = DATA_BLOB(len(raw), ctypes.cast(ctypes.create_string_buffer(raw), ctypes.POINTER(ctypes.c_char)))
        out_blob = DATA_BLOB()
        if not CryptProtectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
            return raw
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(out_blob.pbData)
    except Exception:
        return raw


def _win_dpapi_unprotect(enc: bytes) -> bytes:
    try:
        if os.name != 'nt':
            return enc
        import ctypes
        from ctypes import wintypes
        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
        CryptUnprotectData = ctypes.windll.crypt32.CryptUnprotectData
        CryptUnprotectData.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(DATA_BLOB), wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
        CryptUnprotectData.restype = wintypes.BOOL
        in_blob = DATA_BLOB(len(enc), ctypes.cast(ctypes.create_string_buffer(enc), ctypes.POINTER(ctypes.c_char)))
        out_blob = DATA_BLOB()
        if not CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
            return enc
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(out_blob.pbData)
    except Exception:
        return enc


def _protect_str(s: str) -> str:
    try:
        raw = (s or '').encode('utf-8')
        enc = _win_dpapi_protect(raw)
        return base64.b64encode(enc).decode('ascii')
    except Exception:
        return s or ''


def _unprotect_str(s: str) -> str:
    try:
        raw = base64.b64decode(s.encode('ascii'))
        dec = _win_dpapi_unprotect(raw)
        return dec.decode('utf-8', errors='replace')
    except Exception:
        return s or ''

def _normalize_name(name: str) -> str:
    try:
        return (name or '').lower().replace('-', '').replace('_', '').replace(':', '')
    except Exception:
        return str(name)


def _names_match(a: str, b: str) -> bool:
    na = _normalize_name(a)
    nb = _normalize_name(b)
    return na in nb or nb in na

# Vehicle knowledge base validation
_vehicle_kb_cache = None
_vehicle_kb_path = None

def load_vehicle_knowledge_base(json_path: str = None) -> dict:
    """Load vehicle knowledge base JSON file. Returns empty dict if not found."""
    global _vehicle_kb_cache, _vehicle_kb_path
    
    # Use cached version if path hasn't changed
    if _vehicle_kb_cache is not None and json_path == _vehicle_kb_path:
        return _vehicle_kb_cache
    
    _vehicle_kb_path = json_path
    _vehicle_kb_cache = {}
    
    if not json_path or not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            _vehicle_kb_cache = json.load(f)
        _log(f"Loaded vehicle knowledge base: {len(_vehicle_kb_cache)} makes, {sum(len(models) for models in _vehicle_kb_cache.values())} total models")
    except Exception as e:
        _log(f"Failed to load vehicle knowledge base: {e}")
        _vehicle_kb_cache = {}
    
    return _vehicle_kb_cache

def normalize_model_name(model: str) -> str:
    """Normalize model name for fuzzy matching (e.g., '488 GTB / Pista' -> '488 GTB')"""
    if not model or model.lower() in ('unknown', 'none', 'n/a'):
        return ''
    
    # Remove common suffixes/variants for matching
    model = model.strip()
    # Remove parenthetical info like "(991.2 Generation)"
    model = re.sub(r'\s*\([^)]+\)', '', model)
    # Remove "/" variants (e.g., "488 GTB / Pista" -> "488 GTB")
    if '/' in model:
        model = model.split('/')[0].strip()
    # Remove common suffixes
    model = re.sub(r'\s+(GTB|Pista|Speciale|Italia|Stradale|Tributo|SVJ?|EVO|STO|GT3|Turbo|Carrera|RS[0-9]*)\s*$', '', model, flags=re.IGNORECASE)
    return model.strip()

def find_matching_model(kb: dict, make: str, model: str) -> tuple[dict | None, str]:
    """
    Find matching model in knowledge base. Returns (model_data, notes).
    If not found, returns (None, validation_notes).
    """
    if not kb or not make or not model:
        return None, "Make or Model missing"
    
    make_norm = _normalize_name(make)
    model_norm = normalize_model_name(model)
    
    # Find make in KB
    matching_make = None
    for kb_make in kb.keys():
        if _normalize_name(kb_make) == make_norm:
            matching_make = kb_make
            break
    
    if not matching_make:
        return None, f"Make '{make}' not found in knowledge base"
    
    # Find model in make's models
    models_list = kb.get(matching_make, [])
    if not models_list:
        return None, f"Make '{make}' found but has no models in knowledge base"
    
    # Try exact match first
    for model_entry in models_list:
        kb_model = model_entry.get('model', '')
        if _normalize_name(kb_model) == _normalize_name(model):
            return model_entry, f"Exact match: {matching_make} {kb_model}"
    
    # Try normalized match (without variants)
    for model_entry in models_list:
        kb_model = model_entry.get('model', '')
        kb_model_norm = normalize_model_name(kb_model)
        if kb_model_norm and model_norm and kb_model_norm.lower() == model_norm.lower():
            return model_entry, f"Normalized match: {matching_make} {kb_model} (matched '{model}')"
    
    # Try partial match (e.g., "488" matches "488 GTB")
    for model_entry in models_list:
        kb_model = model_entry.get('model', '')
        kb_model_norm = normalize_model_name(kb_model)
        # Check if model number/name is contained
        if model_norm and kb_model_norm:
            # Extract base model number/name (first word or number)
            model_base = model_norm.split()[0] if model_norm.split() else ''
            kb_base = kb_model_norm.split()[0] if kb_model_norm.split() else ''
            if model_base and kb_base and model_base.lower() == kb_base.lower():
                return model_entry, f"Partial match: {matching_make} {kb_model} (matched base '{model_base}')"
    
    # No match found
    available_models = [m.get('model', '') for m in models_list[:5]]  # Show first 5
    return None, f"Model '{model}' not found for {matching_make}. Available models include: {', '.join(available_models)}"

def validate_vehicle_identification(parsed: dict, kb_path: str = None) -> tuple[bool, str, dict | None]:
    """
    Validate identified Make/Model against knowledge base.
    Returns: (is_valid, validation_notes, model_data)
    - is_valid: True if Make/Model found in KB, False otherwise
    - validation_notes: Human-readable notes about validation
    - model_data: Matching model entry from KB (with distinguishing_cues) or None
    
    Uses three-tier KB system:
    1. Built-in KB (supercar_cheat_sheet.json)
    2. Approved KB (approved_knowledge_base.json) - developer-approved items
    3. User KB (user_knowledge_base.json) - highest priority, never overwritten
    """
    # Skip validation for non-vehicle images
    if parsed.get('Image Type') == 'Non-Car Scene' or 'NC_' in str(parsed.keys()):
        return True, "Non-vehicle image - validation skipped", None
    
    make = parsed.get('Make', '').strip()
    model = parsed.get('Model', '').strip()
    
    if not make or make.lower() in ('unknown', 'none', 'n/a'):
        return False, "Make is missing or Unknown", None
    
    if not model or model.lower() in ('unknown', 'none', 'n/a'):
        return False, "Model is missing or Unknown", None
    
    # Load all knowledge bases
    builtin_kb = load_vehicle_knowledge_base(kb_path)
    approved_kb = load_approved_knowledge_base()
    user_kb = load_user_knowledge_base()
    
    # Merge in priority order: built-in -> approved -> user (user overrides all)
    kb = merge_knowledge_bases(builtin_kb, approved_kb, user_kb)
    
    if not kb:
        return True, "Knowledge base not available - validation skipped", None
    
    # Find matching model
    model_data, notes = find_matching_model(kb, make, model)
    
    if model_data:
        return True, notes, model_data
    else:
        return False, notes, None


# User Knowledge Base Management
_user_kb_cache = None
_user_kb_path = None
_approved_kb_cache = None
_approved_kb_path = None

def load_approved_knowledge_base(json_path: str = None) -> dict:
    """Load developer-approved knowledge base JSON file. Returns empty dict if not found.
    This is for approved items from user submissions that can be included in releases.
    User KB entries take precedence over approved KB.
    
    Checks in order:
    1. Specified path (if provided)
    2. Next to executable (approved_knowledge_base.json)
    3. Bundled resources (if included in exe)
    """
    global _approved_kb_cache, _approved_kb_path
    
    if json_path is None:
        # Try next to exe first
        json_path = os.path.join(_app_base_dir(), 'approved_knowledge_base.json')
        
        # If not found, try bundled resources
        if not os.path.exists(json_path):
            bundled_path = _resource_path('approved_knowledge_base.json')
            if bundled_path and os.path.exists(bundled_path):
                json_path = bundled_path
    
    # Use cached version if path hasn't changed
    if _approved_kb_cache is not None and json_path == _approved_kb_path:
        return _approved_kb_cache
    
    _approved_kb_path = json_path
    _approved_kb_cache = {}
    
    if not json_path or not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            _approved_kb_cache = json.load(f)
        _log(f"Loaded approved knowledge base: {len(_approved_kb_cache)} makes")
    except Exception as e:
        _log(f"Failed to load approved knowledge base: {e}")
        _approved_kb_cache = {}
    
    return _approved_kb_cache

def load_user_knowledge_base(json_path: str = None) -> dict:
    """Load user's custom knowledge base JSON file. Returns empty dict if not found.
    This KB is NEVER overwritten by updates - it has the highest priority."""
    global _user_kb_cache, _user_kb_path
    
    if json_path is None:
        json_path = os.path.join(_app_base_dir(), 'user_knowledge_base.json')
    
    # Use cached version if path hasn't changed
    if _user_kb_cache is not None and json_path == _user_kb_path:
        return _user_kb_cache
    
    _user_kb_path = json_path
    _user_kb_cache = {}
    
    if not json_path or not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            _user_kb_cache = json.load(f)
        _log(f"Loaded user knowledge base: {len(_user_kb_cache)} makes")
    except Exception as e:
        _log(f"Failed to load user knowledge base: {e}")
        _user_kb_cache = {}
    
    return _user_kb_cache

def merge_knowledge_bases(builtin_kb: dict, approved_kb: dict = None, user_kb: dict = None) -> dict:
    """
    Merge knowledge bases in priority order:
    1. Built-in KB (base)
    2. Approved KB (developer-approved items, can be updated in releases)
    3. User KB (highest priority, never overwritten)
    
    User entries override approved entries, which override built-in entries.
    """
    # Start with built-in KB
    if not builtin_kb:
        merged = {}
    else:
        merged = builtin_kb.copy()
    
    # Merge approved KB (if provided)
    if approved_kb:
        for make, models in approved_kb.items():
            if make not in merged:
                merged[make] = []
            
            if isinstance(models, list):
                # Merge models, approved entries override built-in
                existing_models = {m.get('model', ''): i for i, m in enumerate(merged.get(make, []))}
                for approved_model in models:
                    model_name = approved_model.get('model', '')
                    if model_name in existing_models:
                        # Replace existing built-in entry
                        merged[make][existing_models[model_name]] = approved_model
                    else:
                        # Add new approved entry
                        merged[make].append(approved_model)
    
    # Merge user KB (highest priority - overrides everything)
    if user_kb:
        for make, models in user_kb.items():
            if make not in merged:
                merged[make] = []
            
            if isinstance(models, list):
                # Merge models, user entries take precedence over both built-in and approved
                existing_models = {m.get('model', ''): i for i, m in enumerate(merged.get(make, []))}
                for user_model in models:
                    model_name = user_model.get('model', '')
                    if model_name in existing_models:
                        # Replace existing (built-in or approved)
                        merged[make][existing_models[model_name]] = user_model
                    else:
                        # Add new user entry
                        merged[make].append(user_model)
            else:
                # If user KB has dict format, convert to list
                merged[make] = [models] if isinstance(models, dict) else []
    
    return merged

def cleanup_unreferenced_kb_images() -> tuple[int, list[str]]:
    """
    Remove images from kb_references folder that are not referenced in any KB JSON file.
    Returns (deleted_count, deleted_files_list).
    """
    try:
        kb_ref_dir = os.path.join(_app_base_dir(), 'kb_references')
        
        # If folder doesn't exist, nothing to clean
        if not os.path.exists(kb_ref_dir):
            return 0, []
        
        # Collect all reference_image paths from all KB files
        referenced_images = set()
        
        # Load built-in KB
        builtin_kb_path = os.path.join(_app_base_dir(), 'supercar_cheat_sheet.json')
        if os.path.exists(builtin_kb_path):
            builtin_kb = load_vehicle_knowledge_base(builtin_kb_path)
            for make, models in builtin_kb.items():
                if isinstance(models, list):
                    for entry in models:
                        if isinstance(entry, dict):
                            ref_img = entry.get('reference_image', '')
                            if ref_img:
                                referenced_images.add(ref_img)
        
        # Load approved KB
        approved_kb = load_approved_knowledge_base()
        for make, models in approved_kb.items():
            if isinstance(models, list):
                for entry in models:
                    if isinstance(entry, dict):
                        ref_img = entry.get('reference_image', '')
                        if ref_img:
                            referenced_images.add(ref_img)
        
        # Load user KB
        user_kb = load_user_knowledge_base()
        for make, models in user_kb.items():
            if isinstance(models, list):
                for entry in models:
                    if isinstance(entry, dict):
                        ref_img = entry.get('reference_image', '')
                        if ref_img:
                            referenced_images.add(ref_img)
        
        # Normalize referenced image paths to just filenames
        # Handle different path formats: "kb_references/file.jpg", "file.jpg", or full paths
        referenced_filenames = set()
        for ref_path in referenced_images:
            if ref_path:
                # Extract just the filename
                filename = os.path.basename(ref_path)
                # Remove any "kb_references/" prefix if present
                if filename.startswith('kb_references/'):
                    filename = filename.replace('kb_references/', '')
                elif '/' in filename or '\\' in filename:
                    filename = os.path.basename(filename)
                referenced_filenames.add(filename.lower())  # Case-insensitive comparison
        
        # Scan kb_references folder and delete unreferenced files
        deleted_files = []
        deleted_count = 0
        
        try:
            for filename in os.listdir(kb_ref_dir):
                file_path = os.path.join(kb_ref_dir, filename)
                
                # Only process image files
                if not os.path.isfile(file_path):
                    continue
                
                # Check if it's an image file
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'):
                    continue
                
                # Check if this file is referenced in any KB
                if filename.lower() not in referenced_filenames:
                    try:
                        os.remove(file_path)
                        deleted_files.append(filename)
                        deleted_count += 1
                        _log(f"Deleted unreferenced KB image: {filename}")
                    except Exception as e:
                        _log(f"Failed to delete {filename}: {e}")
        
        except Exception as e:
            _log(f"Error scanning kb_references folder: {e}")
        
        if deleted_count > 0:
            _log(f"KB cleanup: Removed {deleted_count} unreferenced image(s)")
        
        return deleted_count, deleted_files
        
    except Exception as e:
        _log(f"Error in cleanup_unreferenced_kb_images: {e}")
        import traceback
        _log(traceback.format_exc())
        return 0, []

def save_to_user_knowledge_base(json_path: str, make: str, model: str, features: dict, reference_image_path: str = None) -> bool:
    """Save a new entry to the user knowledge base JSON file."""
    try:
        # Load existing KB
        kb = load_user_knowledge_base(json_path)
        
        # Ensure make exists
        if make not in kb:
            kb[make] = []
        
        # Check if model already exists
        existing_idx = None
        for i, entry in enumerate(kb[make]):
            if isinstance(entry, dict) and entry.get('model', '').lower() == model.lower():
                existing_idx = i
                break
        
        # Create new entry
        new_entry = {
            'model': model,
            'year': 'Unknown',
            'distinguishing_cues': features,
        }
        
        if reference_image_path:
            new_entry['reference_image'] = reference_image_path
        
        # Update or add entry
        if existing_idx is not None:
            kb[make][existing_idx] = new_entry
        else:
            kb[make].append(new_entry)
        
        # Save to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
        
        # Clear cache to force reload
        global _user_kb_cache
        _user_kb_cache = None
        
        _log(f"Saved to user KB: {make} {model}")
        return True
        
    except Exception as e:
        _log(f"Error saving to user KB: {e}")
        import traceback
        _log(traceback.format_exc())
        return False

def save_reference_image(image_path: str, make: str, model: str) -> str:
    """Save an optimized reference image for the KB entry. Returns path to saved image."""
    try:
        # Create kb_references directory if it doesn't exist
        kb_ref_dir = os.path.join(_app_base_dir(), 'kb_references')
        os.makedirs(kb_ref_dir, exist_ok=True)
        
        # Load and optimize image
        img = Image.open(image_path)
        w, h = img.size
        
        # Resize to 512x512 max (maintain aspect ratio)
        max_size = 512
        if w > max_size or h > max_size:
            if w >= h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Generate filename: Make_Model_hash.jpg
        import hashlib
        hash_str = hashlib.md5(image_path.encode()).hexdigest()[:8]
        safe_make = make.replace(' ', '_').replace('/', '_')
        safe_model = model.replace(' ', '_').replace('/', '_')
        filename = f"{safe_make}_{safe_model}_{hash_str}.jpg"
        ref_path = os.path.join(kb_ref_dir, filename)
        
        # Save as JPEG
        img.save(ref_path, 'JPEG', quality=85, optimize=True)
        
        _log(f"Saved reference image: {ref_path}")
        return ref_path
        
    except Exception as e:
        _log(f"Error saving reference image: {e}")
        return ''

def extract_distinguishing_features_ai(client, model_name: str, image_path: str, make: str, model: str) -> dict:
    """Use AI to extract distinguishing visual features from the image."""
    try:
        # Encode image
        image_b64 = encode_image_jpeg_b64(image_path, max_size=2048, quality=85)
        
        # Create prompt for feature extraction
        prompt = f"""Analyze this {make} {model} and extract distinguishing visual features that would help identify this specific model.

Focus on:
- Front: grille shape, headlight design, bumper style, logo placement
- Side: body shape, door design, window shape, side vents/intakes
- Rear: taillight design, exhaust position/style, diffuser, spoiler
- Grille: pattern, shape, logo placement
- Lights: headlight and taillight distinctive features
- Logo: placement and style

Return ONLY a JSON object with these keys:
{{
  "front": "description of front features",
  "side": "description of side profile features",
  "rear": "description of rear features",
  "grille": "description of grille features",
  "lights": "description of headlight/taillight features",
  "logo": "description of logo placement and style"
}}

Be specific and detailed. Focus on features that distinguish this model from similar models."""
        
        # Call LLM
        response = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_b64]
                }
            ]
        )
        
        # Extract text
        text = extract_message_text(response)
        
        # Try to parse JSON
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"front"[^{}]*\}', text, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group(0))
            else:
                # Fallback: try to parse entire response as JSON
                features = json.loads(text)
        except:
            # Fallback: create structured features from text
            features = {
                'front': text[:200] if text else 'Features extracted from image',
                'side': '',
                'rear': '',
                'grille': '',
                'lights': '',
                'logo': ''
            }
        
        return features
        
    except Exception as e:
        _log(f"Error in extract_distinguishing_features_ai: {e}")
        import traceback
        _log(traceback.format_exc())
        return {
            'front': '',
            'side': '',
            'rear': '',
            'grille': '',
            'lights': '',
            'logo': ''
        }

def compare_with_reference_image(client, model_name: str, current_image_path: str, reference_image_path: str) -> float:
    """Use AI to compare current image with reference image and return similarity score (0-1)."""
    try:
        # Encode both images
        current_b64 = encode_image_jpeg_b64(current_image_path, max_size=2048, quality=85)
        reference_b64 = encode_image_jpeg_b64(reference_image_path, max_size=2048, quality=85)
        
        prompt = """Compare these two images of vehicles. Are they the same make and model?

Look for:
- Similar body shape and proportions
- Similar grille design
- Similar headlight/taillight design
- Similar overall styling

Respond with ONLY a number between 0.0 and 1.0 where:
- 1.0 = Definitely the same make and model
- 0.8-0.9 = Very likely the same
- 0.6-0.7 = Possibly the same
- 0.4-0.5 = Unlikely the same
- 0.0-0.3 = Definitely different

Return ONLY the number, nothing else."""
        
        response = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [current_b64, reference_b64]
                }
            ]
        )
        
        text = extract_message_text(response)
        
        # Extract number from response
        score_match = re.search(r'0?\.\d+|1\.0|\d+', text)
        if score_match:
            score = float(score_match.group(0))
            return min(1.0, max(0.0, score))
        
        return 0.5  # Default neutral score
        
    except Exception as e:
        _log(f"Error in compare_with_reference_image: {e}")
        return 0.5

def export_knowledge_base_for_submission(output_path: str) -> bool:
    """Export user knowledge base as an encrypted ZIP file for submission to developer."""
    try:
        import zipfile
        import tempfile
        import shutil
        
        # Load user KB
        user_kb_path = os.path.join(_app_base_dir(), 'user_knowledge_base.json')
        if not os.path.exists(user_kb_path):
            return False
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Copy KB file
            kb_copy = os.path.join(temp_dir, 'user_knowledge_base.json')
            shutil.copy2(user_kb_path, kb_copy)
            
            # Copy reference images
            kb_ref_dir = os.path.join(_app_base_dir(), 'kb_references')
            if os.path.exists(kb_ref_dir):
                ref_copy_dir = os.path.join(temp_dir, 'kb_references')
                shutil.copytree(kb_ref_dir, ref_copy_dir)
            
            # Create ZIP
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arc_name)
            
            # Encrypt ZIP with simple XOR (for basic obfuscation)
            # In production, use stronger encryption
            with open(output_path, 'rb') as f:
                data = f.read()
            
            # Simple XOR encryption with key
            key = b'TAGOMATIC_KB_EXPORT_2024'
            encrypted = bytearray()
            for i, byte in enumerate(data):
                encrypted.append(byte ^ key[i % len(key)])
            
            with open(output_path, 'wb') as f:
                f.write(bytes(encrypted))
            
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        _log(f"Error exporting KB: {e}")
        import traceback
        _log(traceback.format_exc())
        return False

def decrypt_knowledge_base_submission(encrypted_path: str, output_dir: str) -> bool:
    """Decrypt a knowledge base submission file."""
    try:
        import zipfile
        import shutil
        
        # Read encrypted file
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt (XOR with same key)
        key = b'TAGOMATIC_KB_EXPORT_2024'
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ key[i % len(key)])
        
        # Write to temp file
        import tempfile
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.write(bytes(decrypted))
        temp_zip.close()
        
        try:
            # Extract ZIP
            with zipfile.ZipFile(temp_zip.name, 'r') as zipf:
                zipf.extractall(output_dir)
            
            return True
            
        finally:
            os.unlink(temp_zip.name)
            
    except Exception as e:
        _log(f"Error decrypting KB submission: {e}")
        return False


def read_image_datetime(image_path: str):
    """Read EXIF DateTimeOriginal from image file."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        from datetime import datetime
        
        img = Image.open(image_path)
        exif = img.getexif()
        
        # Look for DateTimeOriginal (tag 306)
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                try:
                    return datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                except Exception:
                    pass
        
        # Fallback: try EXIF DateTime (tag 306)
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTime':
                try:
                    return datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                except Exception:
                    pass
        
        return None
    except Exception as e:
        _log(f"Error reading image datetime: {e}")
        return None


def match_image_to_session(image_path: str, sessions: dict):
    """Match image EXIF DateTimeOriginal to session time ranges from PDF.
    
    Returns:
        str | None: Session name if matched, None otherwise
    """
    img_datetime = read_image_datetime(image_path)
    if not img_datetime:
        return None
    
    # Find session that contains this timestamp
    for session_name, (start_dt, end_dt) in sessions.items():
        if start_dt <= img_datetime <= end_dt:
            return session_name
    
    return None


def match_race_number_to_data(race_number: str, race_data: dict):
    """Match race number to PDF race data.
    
    Returns:
        dict | None: Race data dict (car, team, driver, session) if found, None otherwise
    """
    # Try exact match first
    if race_number in race_data:
        return race_data[race_number]
    
    # Try case-insensitive match
    race_number_lower = race_number.lower()
    for num, data in race_data.items():
        if num.lower() == race_number_lower:
            return data
    
    return None


def match_race_number_with_car(race_number: str, ai_make: str, ai_model: str, ai_color: str, race_data: dict) -> dict | None:
    """Match race number to PDF data using car Make/Model/Color when multiple matches exist.
    
    This is used when the same race number appears in different classes (e.g., GT3 #79 and GT4 #79).
    Uses AI-identified Make/Model/Color to find the best match.
    
    Args:
        race_number: Race number to match
        ai_make: AI-identified make (e.g., "Nissan")
        ai_model: AI-identified model (e.g., "Micra")
        ai_color: AI-identified color (e.g., "blue")
        race_data: Filtered race data dict (already filtered by session)
    
    Returns:
        dict | None: Best matching race data entry, or None if no match
    """
    # Find all entries matching race number
    matches = []
    race_number_lower = race_number.lower()
    
    for num, data in race_data.items():
        if num.lower() == race_number_lower:
            matches.append((num, data))
    
    if not matches:
        return None
    
    # If only one match, return it
    if len(matches) == 1:
        return matches[0][1]
    
    # Multiple matches - use car matching
    # Normalize AI values for comparison (functions defined earlier in file)
    try:
        ai_make_norm = _normalize_name(ai_make) if ai_make else ''
    except NameError:
        ai_make_norm = (ai_make or '').lower().strip()
    
    try:
        ai_model_norm = normalize_model_name(ai_model) if ai_model else ''
    except NameError:
        ai_model_norm = (ai_model or '').lower().strip()
    ai_color_norm = (ai_color or '').lower().strip()
    
    best_match = None
    best_score = 0
    
    for num, data in matches:
        pdf_car = (data.get('car') or '').strip()
        pdf_car_upper = pdf_car.upper()
        
        score = 0
        
        # Check Make match
        if ai_make_norm:
            # Check if make appears in PDF car description
            for make_word in ai_make.split():
                if make_word.upper() in pdf_car_upper:
                    score += 3  # Make match is important
                    break
        
        # Check Model match
        if ai_model_norm:
            # Try normalized model name
            model_words = ai_model_norm.split()
            for word in model_words:
                if len(word) >= 3 and word.upper() in pdf_car_upper:
                    score += 2  # Model match is important
                    break
        
        # Check Color match (fuzzy)
        if ai_color_norm:
            color_variants = {
                'blue': ['blue', 'blu', 'azure', 'cyan'],
                'red': ['red', 'crimson', 'scarlet', 'maroon'],
                'green': ['green', 'emerald', 'lime'],
                'black': ['black', 'dark', 'charcoal'],
                'white': ['white', 'silver', 'pearl'],
                'yellow': ['yellow', 'gold', 'amber'],
            }
            
            for base_color, variants in color_variants.items():
                if base_color in ai_color_norm:
                    for variant in variants:
                        if variant in pdf_car_upper.lower():
                            score += 1  # Color match is helpful but less critical
                            break
                    break
        
        # Update best match
        if score > best_score:
            best_score = score
            best_match = data
    
    # Return best match, or first match if no car-based match found
    return best_match if best_match else matches[0][1]


def _get_ollama_base_url(client) -> str:
    try:
        base = getattr(client, 'host', None) or getattr(client, 'base_url', None)
        if not base:
            base = 'http://localhost:11434'
        return str(base).rstrip('/')
    except Exception:
        return 'http://localhost:11434'


def list_models_via_http(client) -> list[str]:
    import json as _json
    from urllib import request
    def fetch(url):
        req = request.Request(url, method='GET')
        with request.urlopen(req, timeout=5) as resp:
            data = resp.read()
            parsed = _json.loads(data.decode('utf-8'))
            models = parsed.get('models', []) if isinstance(parsed, dict) else []
            names_local = []
            for m in models:
                if isinstance(m, dict):
                    nm = m.get('name') or m.get('model') or m.get('tag')
                    if nm:
                        names_local.append(nm)
            return names_local
    bases = [_get_ollama_base_url(client), 'http://127.0.0.1:11434']
    for base in bases:
        try:
            names = fetch(f"{base}/api/tags")
            if names:
                return names
        except Exception:
            continue
    return []


def pick_preferred_vision_model(names: list[str], default: str) -> str:
    if not names:
        return default
    preferences = ['qwen2.5vl', 'qwen-2.5-vl', 'vl', 'vision', 'llava', 'bakllava', 'pixtral', 'moondream', 'minicpm', 'phi-3.5-vision', 'llama3.2-vision', 'llama-3.2-vision', 'qwen-vl']
    for pref in preferences:
        for n in names:
            if pref in n.lower():
                return n
    return names[0]


def optimize_image_for_ollama(image_path: str) -> str:
    """Simple, fast image optimization like the working backup"""
    try:
        img = Image.open(image_path)
        try:
            img.draft('RGB', (2048, 2048))
        except Exception:
            pass
        w, h = img.size
        min_required = 28
        max_size = 2048
        need_up = (w > 0 and h > 0 and min(w, h) < min_required)
        need_down = w > max_size or h > max_size
        if not need_up and not need_down:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        if need_up:
            if w < h:
                new_w = min_required
                new_h = int(round(h * (min_required / w)))
            else:
                new_h = min_required
                new_w = int(round(w * (min_required / h)))
            img = img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
        elif need_down:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90, optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


def encode_image_png_b64(image_path: str, max_size: int = 2048) -> str:
    """Robustly load and re-encode as PNG to recover from truncated/invalid JPEGs."""
    try:
        img = Image.open(image_path)
        try:
            img.load()
        except Exception:
            pass
        w, h = img.size
        if w > max_size or h > max_size:
            if w >= h:
                new_w = max_size
                new_h = int(h * max_size / max(1, w))
            else:
                new_h = max_size
                new_w = int(w * max_size / max(1, h))
            img = img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


def encode_image_jpeg_b64(image_path: str, max_size: int = 2048, quality: int = 80) -> str:
    """Cloud-safe JPEG encoding: resize to <= max_size on long side and compress.
    Returns base64 JPEG string.
    """
    try:
        img = Image.open(image_path)
        try:
            img.load()
        except Exception:
            pass
        w, h = img.size
        if w > max_size or h > max_size:
            if w >= h:
                new_w = max_size
                new_h = int(h * max_size / max(1, w))
            else:
                new_h = max_size
                new_w = int(w * max_size / max(1, h))
            img = img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=max(80, min(95, int(quality))), optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

def extract_message_text(response) -> str:
    try:
        # Safety check: handle None or invalid response
        if response is None:
            print(f"[DEBUG] extract_message_text: response is None, returning empty string")
            return ''
        
        print(f"[DEBUG] extract_message_text: FUNCTION START - response type={type(response).__name__}")
        # Debug: log the response type and structure
        response_type = type(response).__name__
        print(f"[DEBUG] extract_message_text: response type={response_type}")
        
        # Safe dir() call - might fail for some objects
        try:
            print(f"[DEBUG] extract_message_text: response dir={dir(response)[:15]}")
        except Exception:
            print(f"[DEBUG] extract_message_text: could not get dir() for response")
        
        if hasattr(response, 'message'):
            try:
                print(f"[DEBUG] extract_message_text: response.message exists, type={type(response.message).__name__}")
                print(f"[DEBUG] extract_message_text: response.message dir={dir(response.message)[:15]}")
            except Exception:
                print(f"[DEBUG] extract_message_text: response.message exists but could not inspect")
        print(f"[DEBUG] extract_message_text: checking if response is tuple")
        if isinstance(response, tuple) and len(response) >= 1:
            print(f"[DEBUG] extract_message_text: response is tuple, extracting first element")
            response = response[0]
        print(f"[DEBUG] extract_message_text: checking if response is iterable")
        if hasattr(response, '__iter__') and not isinstance(response, (dict, str, bytes, list, tuple)) and not hasattr(response, 'message'):
            print(f"[DEBUG] extract_message_text: response is iterable, processing chunks")
            combined = ''
            for chunk in response:
                if isinstance(chunk, dict):
                    msg = chunk.get('message')
                    if isinstance(msg, dict):
                        c = msg.get('content')
                        if isinstance(c, str):
                            combined += c
                elif isinstance(chunk, str):
                    combined += chunk
            print(f"[DEBUG] extract_message_text: returning combined content from iterable")
            return combined
        print(f"[DEBUG] extract_message_text: checking if response is dict")
        if isinstance(response, dict) and not hasattr(response, 'message'):
            print(f"[DEBUG] extract_message_text: response is dict, checking for content")
            if 'message' in response and isinstance(response['message'], dict):
                c = response['message'].get('content')
                if isinstance(c, str):
                    print(f"[DEBUG] extract_message_text: found content in dict['message'], returning")
                    return c
            if 'messages' in response and isinstance(response['messages'], list):
                for msg in response['messages']:
                    if isinstance(msg, dict) and msg.get('role') == 'assistant':
                        c = msg.get('content')
                        if isinstance(c, str):
                            print(f"[DEBUG] extract_message_text: found content in dict['messages'], returning")
                            return c
            for key in ('content', 'text', 'output'):
                c = response.get(key)
                if isinstance(c, str):
                    print(f"[DEBUG] extract_message_text: found content in dict[{key}], returning")
                    return c
            print(f"[DEBUG] extract_message_text: dict response has no content, continuing")
            return ''
        print(f"[DEBUG] extract_message_text: past dict handling, checking message attributes")
        # Handle Ollama response objects with message attributes
        print(f"[DEBUG] extract_message_text: checking if response has message attribute")
        print(f"[DEBUG] extract_message_text: about to enter message handling section")
        print(f"[DEBUG] extract_message_text: response type is {type(response).__name__}")
        if hasattr(response, 'message'):
            print(f"[DEBUG] extract_message_text: response has message attribute, message type={type(response.message).__name__}")
            print(f"[DEBUG] extract_message_text: message dir contains: {[attr for attr in dir(response.message) if not attr.startswith('_')]}")
            
            # Try multiple ways to access content
            content_found = False
            
            # Method 1: Direct attribute access
            try:
                c = response.message.content
                print(f"[DEBUG] extract_message_text: Method 1 - direct access: content type={type(c).__name__}, value={str(c)[:50] if c else 'None'}")
                if isinstance(c, str) and c.strip():
                    print(f"[DEBUG] extract_message_text: Method 1 SUCCESS - returning content, length={len(c)}")
                    return c
                content_found = True
            except Exception as e:
                print(f"[DEBUG] extract_message_text: Method 1 failed: {e}")
            
            # Method 2: Dictionary access
            try:
                if hasattr(response.message, '__dict__'):
                    print(f"[DEBUG] extract_message_text: Method 2 - __dict__ keys: {list(response.message.__dict__.keys())}")
                    if 'content' in response.message.__dict__:
                        c = response.message.__dict__['content']
                        print(f"[DEBUG] extract_message_text: Method 2 - found content in __dict__, type={type(c).__name__}")
                        if isinstance(c, str) and c.strip():
                            print(f"[DEBUG] extract_message_text: Method 2 SUCCESS - returning content, length={len(c)}")
                            return c
                        content_found = True
            except Exception as e:
                print(f"[DEBUG] extract_message_text: Method 2 failed: {e}")
            
            # Method 3: Getattr access
            try:
                c = getattr(response.message, 'content', None)
                if c is not None:
                    print(f"[DEBUG] extract_message_text: Method 3 - getattr result: type={type(c).__name__}, value={str(c)[:50] if c else 'None'}")
                    if isinstance(c, str) and c.strip():
                        print(f"[DEBUG] extract_message_text: Method 3 SUCCESS - returning content, length={len(c)}")
                        return c
                    content_found = True
            except Exception as e:
                print(f"[DEBUG] extract_message_text: Method 3 failed: {e}")

            # Do NOT return chain-of-thought (message.thinking). We intentionally avoid
            # leaking planning text into summaries/metadata. If content is empty, the
            # caller should retry with a stricter prompt instead.
            
            if not content_found:
                print(f"[DEBUG] extract_message_text: All methods failed to find content")
        print(f"[DEBUG] extract_message_text: finished message handling section")
        print(f"[DEBUG] extract_message_text: checking if response is string")
        try:
            if isinstance(response, str):
                print(f"[DEBUG] extract_message_text: response is string, returning it")
                return response
        except Exception as e:
            print(f"[DEBUG] extract_message_text: isinstance() check failed: {e}")
            # If isinstance fails, response is corrupted - return empty string
            return ''
        
        print(f"[DEBUG] extract_message_text: no content found, returning empty string")
        print(f"[DEBUG] extract_message_text: FUNCTION END - returning empty string")
        return ''
    except AttributeError as e:
        print(f"[DEBUG] extract_message_text: AttributeError: {e}, response type={type(response).__name__ if response else 'None'}")
        import traceback
        print(f"[DEBUG] extract_message_text: AttributeError traceback: {traceback.format_exc()}")
        return ''
    except Exception as e:
        print(f"[DEBUG] extract_message_text: EXCEPTION CAUGHT: {type(e).__name__}: {e}")
        import traceback
        print(f"[DEBUG] extract_message_text: Exception traceback: {traceback.format_exc()}")
        return ''


def extract_message_text_strict(response, allow_reasoning: bool = False) -> str:
    """Prefer message.content; optionally fall back to message.thinking when allowed.

    This is used for internal routing/classification only when Qwen-3VL emits
    empty content but provides usable short answers in thinking. Never use the
    reasoning text for user-visible metadata.
    """
    try:
        txt = extract_message_text(response)
        if isinstance(txt, str) and txt.strip():
            return txt
        if not allow_reasoning:
            return ''
        try:
            if hasattr(response, 'message'):
                t = getattr(response.message, 'thinking', None)
                if isinstance(t, str):
                    return t.strip()
        except Exception:
            pass
        return ''
    except Exception:
        return ''


def extract_message_thinking(response) -> str:
    try:
        if hasattr(response, 'message'):
            t = getattr(response.message, 'thinking', None)
            if isinstance(t, str):
                return t
        return ''
    except Exception:
        return ''


def _looks_meta_only_json(text: str) -> bool:
    try:
        s = (text or '').strip()
        if not (s.startswith('{') and s.endswith('}')):
            return False
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return False
        keys = set(k.lower() for k in obj.keys())
        allowed = {'finishreason', 'responseid', 'modelversion', 'usagemetadata', 'usage', 'prompttokencount', 'totaltokencount'}
        return keys.issubset(allowed)
    except Exception:
        return False

def parse_results_lines(result_text: str) -> dict:
    data = {}
    lines = (result_text or '').split('\n')
    def norm_key(k: str) -> str:
        kl = (k or '').strip().strip(':').lower()
        if 'make' in kl or 'brand' in kl:
            return 'Make'
        if 'model' in kl:
            return 'Model'
        if 'colour' in kl or 'color' in kl:
            return 'Color'
        if 'license' in kl or 'licence' in kl or 'plate' in kl:
            return 'License Plate'
        if 'logo' in kl or 'emblem' in kl or 'badge' in kl or 'text' in kl:
            return 'Logos'
        if 'race' in kl and 'number' in kl:
            return 'Race Number'
        if 'summary' in kl or 'interpretation' in kl:
            return 'AI-Interpretation Summary'
        return k.strip(':')
    for ln in lines:
        if ':' not in ln:
            continue
        k, v = ln.split(':', 1)
        k = norm_key(k)
        v = v.strip().strip('"').strip()
        if v.startswith('**'):
            v = v.lstrip('* ').strip()
        if v.endswith('**'):
            v = v.rstrip('* ').strip()
        if v.lower() in {'unknown', 'not visible', 'unclear', 'none'}:
            continue
        data[k] = v
    if 'AI-Interpretation Summary' not in data:
        head = (result_text or '').strip().split('\n')[0] if result_text else 'Image analysis completed'
        if head.strip().startswith('{'):
            data['AI-Interpretation Summary'] = 'Image analysis completed'
        else:
            data['AI-Interpretation Summary'] = head[:500]  # Increased from 200 to 500 chars
    return data


def parse_or_fallback_json(text: str) -> dict:
    try:
        tp = (text or '').lower()
        if 'make:' in tp and 'model:' in tp:
            data = parse_results_lines(text)
            if data:
                return data
    except Exception:
        pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            import re as _re
            norm = {}
            for k, v in obj.items():
                kk = str(k).lower()
                # Exclude transport/meta keys from mapping
                if kk in {'finishreason', 'responseid', 'modelversion', 'usage', 'usage_metadata', 'usageMetadata'}:
                    continue
                # Map only whole-word matches to avoid 'modelVersion' -> 'Model'
                if _re.search(r'(^|[^a-z])make([^a-z]|$)', kk) or 'brand' in kk:
                    nk = 'Make'
                elif _re.search(r'(^|[^a-z])model([^a-z]|$)', kk):
                    nk = 'Model'
                elif 'color' in kk or 'colour' in kk:
                    nk = 'Color'
                elif 'license' in kk or 'licence' in kk or _re.search(r'(^|[^a-z])plate([^a-z]|$)', kk):
                    nk = 'License Plate'
                elif 'logo' in kk or 'emblem' in kk or 'badge' in kk:
                    nk = 'Logos'
                elif 'race' in kk and 'number' in kk:
                    nk = 'Race Number'
                elif 'summary' in kk or 'interpretation' in kk:
                    nk = 'AI-Interpretation Summary'
                else:
                    continue
                norm[nk] = str(v).strip()
            # CRITICAL FIX: Don't force car fields for non-car images
            # Only set fields that actually exist in the response
            pass
            return norm
    except Exception:
        pass
    data = parse_results_lines(text)
    for k in ['Make', 'Model', 'Color', 'Logos', 'Race Number', 'License Plate', 'AI-Interpretation Summary']:
        data.setdefault(k, 'Unknown' if k != 'Logos' else '')
    return data


# ONNX Runtime model for intelligent vehicle detection (lazy loaded)
# Using ONNX instead of YOLO to avoid PyTorch threading issues with Qt
_onnx_model = None
_onnx_session = None
_onnx_available = False
_onnx_lock = threading.Lock()  # Thread safety for ONNX calls (should be thread-safe, but keeping for safety)
_pil_lock = threading.Lock()  # Thread safety for PIL operations

def _get_onnx_model():
    """Lazy load ONNX model for vehicle detection (thread-safe alternative to YOLO)"""
    global _onnx_model, _onnx_session, _onnx_available
    
    if _onnx_session is not None:
        return _onnx_session
    
    try:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image
        
        # Try to find ONNX model files (check both app directory and PyInstaller _MEIPASS)
        search_dirs = []
        # PyInstaller onefile mode: check _MEIPASS first
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            search_dirs.append(sys._MEIPASS)
        # Regular app directory
        search_dirs.append(_app_base_dir())
        
        onnx_path = None
        pt_path = None
        
        # Check for existing ONNX models first (prefer YOLOv10n)
        for search_dir in search_dirs:
            yolo10_onnx = os.path.join(search_dir, 'yolov10n.onnx')
            yolo8_onnx = os.path.join(search_dir, 'yolov8n.onnx')
            if os.path.exists(yolo10_onnx):
                onnx_path = yolo10_onnx
                _log(f"Found YOLOv10n ONNX model: {onnx_path}")
                break
            elif os.path.exists(yolo8_onnx):
                onnx_path = yolo8_onnx
                _log(f"Found YOLOv8n ONNX model: {onnx_path}")
                break
        
        # If no ONNX, try to convert from PyTorch
        if not onnx_path:
            for search_dir in search_dirs:
                yolo10_pt = os.path.join(search_dir, 'yolov10n.pt')
                yolo8_pt = os.path.join(search_dir, 'yolov8n.pt')
                if os.path.exists(yolo10_pt):
                    pt_path = yolo10_pt
                    _log(f"Found YOLOv10n PyTorch model, will convert to ONNX: {pt_path}")
                    break
                elif os.path.exists(yolo8_pt):
                    pt_path = yolo8_pt
                    _log(f"Found YOLOv8n PyTorch model, will convert to ONNX: {pt_path}")
                    break
        
        # Convert PyTorch to ONNX if needed
        if pt_path and not onnx_path:
            try:
                from ultralytics import YOLO
                _log(f"Converting {pt_path} to ONNX format...")
                yolo_model = YOLO(pt_path)
                onnx_path = pt_path.replace('.pt', '.onnx')
                # Export to ONNX with NMS included (outputs [x1, y1, x2, y2, conf, class])
                # simplify=True optimizes the model, NMS is included by default in ONNX export
                yolo_model.export(format='onnx', imgsz=640, simplify=True)
                if os.path.exists(onnx_path):
                    _log(f"Successfully converted to ONNX: {onnx_path}")
                else:
                    _log(f"ONNX conversion failed - file not created")
                    return None
            except Exception as e:
                _log(f"ONNX conversion failed: {e}")
                return None
        
        if not onnx_path or not os.path.exists(onnx_path):
            _log("No ONNX model found and conversion failed")
            return None
        
        # Load ONNX model with ONNX Runtime
        # Use CPU execution provider (thread-safe, no CUDA threading issues)
        providers = ['CPUExecutionProvider']
        try:
            # Try CUDA if available (but ONNX Runtime handles threading better)
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                _log("ONNX Runtime: CUDA available, will use GPU")
            else:
                _log("ONNX Runtime: Using CPU only")
        except:
            pass
        
        _onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        _onnx_model = onnx_path
        _onnx_available = True
        
        # Test with dummy image
        try:
            test_img = np.array(Image.new('RGB', (640, 640), color='black')).astype(np.float32)
            test_img = test_img.transpose(2, 0, 1)  # HWC to CHW
            test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension
            test_img = test_img / 255.0  # Normalize to [0, 1]
            
            # Get input name
            input_name = _onnx_session.get_inputs()[0].name
            _ = _onnx_session.run(None, {input_name: test_img})
            _log(f"ONNX model loaded and tested: {onnx_path}")
            return _onnx_session
        except Exception as test_e:
            _log(f"ONNX test failed: {test_e}")
            _onnx_session = None
            return None
            
    except ImportError:
        _log("onnxruntime not installed. Install with: pip install onnxruntime")
        _onnx_available = False
        return None
    except Exception as e:
        _log(f"ONNX not available: {e}")
        import traceback
        _log(f"ONNX error traceback: {traceback.format_exc()}")
        _onnx_available = False
        return None


def generate_detail_crops_onnx(image_path: str, for_cloud: bool = False) -> list[str]:
    """Generate intelligent crops using ONNX Runtime vehicle detection (thread-safe)
    
    Detects vehicles and generates crops for the dominant (largest) vehicle only.
    Generates: full crop + left side + right side (3 crops total).
    """
    crops: list[str] = []
    
    session = _get_onnx_model()
    if session is None:
        _log("ONNX model unavailable, using fallback geometric crops")
        return []
    
    # Protect PIL operations with lock
    with _pil_lock:
        try:
            import numpy as np
            from PIL import Image
            
            img = Image.open(image_path)
            w, h = img.size
            
            # Early exit for small images
            if w < 512 or h < 512:
                img.close()
                return crops
            
            # Preprocess image for ONNX model (YOLO format: 640x640, RGB, normalized)
            img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32)
            img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            # Run ONNX inference (thread-safe, no PyTorch issues)
            try:
                with _onnx_lock:
                    input_name = session.get_inputs()[0].name
                    outputs = session.run(None, {input_name: img_array})
            except Exception as e:
                _log(f"ONNX inference error: {e}")
                import traceback
                _log(f"ONNX traceback: {traceback.format_exc()}")
                img.close()
                return []
            
            # Process ONNX output (format depends on YOLO export)
            # Standard YOLO ONNX output: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
            # Or raw format: [batch, num_detections, 84] where 84 = 4 (bbox) + 1 (conf) + 79 (classes)
            output = outputs[0]  # Get first output
            if output is None or len(output) == 0:
                _log(f"ONNX: No detections in {image_path}")
                img.close()
                return []
            
            # Vehicle class indices in COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
            vehicle_classes = [2, 3, 5, 7]
            
            # Collect vehicle detections
            vehicles = []
            try:
                # Handle different output formats
                if output.shape[-1] == 6:
                    # Post-processed format: [x1, y1, x2, y2, conf, class]
                    detections = output[0]  # Remove batch dimension
                    for det in detections:
                        if len(det) < 6:
                            continue
                        x1, y1, x2, y2, conf, cls = det[0], det[1], det[2], det[3], det[4], int(det[5])
                        # Skip invalid detections (zero boxes or low confidence)
                        if conf < 0.25 or (x2 - x1) < 1 or (y2 - y1) < 1:
                            continue
                        if cls in vehicle_classes:
                            # Coordinates are in 640x640 space, scale to original image size
                            x1 = int(x1 * w / 640)
                            y1 = int(y1 * h / 640)
                            x2 = int(x2 * w / 640)
                            y2 = int(y2 * h / 640)
                            # Ensure coordinates are within image bounds
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            area = (x2 - x1) * (y2 - y1)
                            vehicles.append({
                                'box': [x1, y1, x2, y2],
                                'area': area,
                                'conf': float(conf)
                            })
                elif output.shape[-1] == 84:
                    # Raw format: need to apply NMS and extract boxes
                    # This is more complex - for now, use simple thresholding
                    detections = output[0]  # Remove batch dimension
                    for det in detections:
                        # Extract bbox (center_x, center_y, width, height) - normalized
                        cx, cy, w_norm, h_norm = det[0], det[1], det[2], det[3]
                        conf = det[4]
                        # Get class scores (indices 5-84)
                        class_scores = det[5:]
                        cls = int(np.argmax(class_scores))
                        max_class_conf = float(np.max(class_scores))
                        
                        if cls in vehicle_classes and conf > 0.25 and max_class_conf > 0.25:
                            # Convert from center+size to xyxy, scale to original image
                            x1 = int((cx - w_norm/2) * w)
                            y1 = int((cy - h_norm/2) * h)
                            x2 = int((cx + w_norm/2) * w)
                            y2 = int((cy + h_norm/2) * h)
                            area = (x2 - x1) * (y2 - y1)
                            vehicles.append({
                                'box': [x1, y1, x2, y2],
                                'area': area,
                                'conf': float(conf * max_class_conf)
                            })
            except Exception as e:
                _log(f"ONNX results processing error: {e}")
                import traceback
                _log(f"ONNX results traceback: {traceback.format_exc()}")
                img.close()
                return []
            
            if not vehicles:
                # No vehicles detected, fallback to geometric
                _log(f"ONNX: No vehicles detected in {image_path}, using geometric fallback")
                img.close()
                return []
            
            # Sort by area (largest first) and take only the dominant (largest) vehicle
            vehicles.sort(key=lambda x: x['area'], reverse=True)
            dominant_vehicle = vehicles[0] if vehicles else None
            
            if not dominant_vehicle:
                _log(f"ONNX: No vehicle to crop in {image_path}")
                img.close()
                return []
            
            # Check if image is already a crop/close-up (vehicle takes up >75% of image)
            # If so, skip cropping to avoid cropping an already-cropped image
            vehicle_area = dominant_vehicle['area']
            image_area = w * h
            coverage = vehicle_area / image_area if image_area > 0 else 0
            if coverage > 0.75:
                _log(f"ONNX: Image appears to be already cropped (vehicle covers {coverage:.1%}), skipping crop generation")
                img.close()
                return []
            
            def encode_crop(crop_img):
                """Helper to encode a crop image to base64"""
                crop_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                if for_cloud:
                    if crop_img.mode != 'RGB':
                        crop_img = crop_img.convert('RGB')
                    crop_img.save(buf, format='JPEG', quality=85, optimize=True)
                else:
                    crop_img.save(buf, format='PNG', optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                return b64
            
            # Generate crops for dominant vehicle only: full + left + right = 3 crops
            x1, y1, x2, y2 = dominant_vehicle['box']
            # Add 10% padding for full vehicle crop
            pad = 0.10
            pw = int(pad * (x2 - x1))
            ph = int(pad * (y2 - y1))
            vx1 = max(0, int(x1) - pw)
            vy1 = max(0, int(y1) - ph)
            vx2 = min(w, int(x2) + pw)
            vy2 = min(h, int(y2) + ph)
            
            vehicle_width = vx2 - vx1
            vehicle_height = vy2 - vy1
            
            # Crop 1: Full vehicle with padding
            full_crop = img.crop((vx1, vy1, vx2, vy2))
            crops.append(encode_crop(full_crop))
            
            # Crop 2: Left side (left 40% of vehicle) - captures one end clearly
            left_x2 = vx1 + int(0.4 * vehicle_width)
            left_crop = img.crop((vx1, vy1, left_x2, vy2))
            crops.append(encode_crop(left_crop))
            
            # Crop 3: Right side (right 40% of vehicle) - captures other end clearly
            right_x1 = vx2 - int(0.4 * vehicle_width)
            right_crop = img.crop((right_x1, vy1, vx2, vy2))
            crops.append(encode_crop(right_crop))
            
            _log(f"ONNX: Detected {len(vehicles)} vehicle(s), cropped dominant vehicle in {image_path}")
            img.close()
        except Exception as e:
            _log(f"ONNX crop error: {e}")
            import traceback
            _log(f"ONNX crop traceback: {traceback.format_exc()}")
            try:
                img.close()
            except:
                pass
            return []
    
    return crops


def generate_detail_crops_geometric(image_path: str, for_cloud: bool = False) -> list[str]:
    """Fallback geometric crops (original method)"""
    crops: list[str] = []
    
    # Protect PIL operations with lock
    with _pil_lock:
        try:
            img = Image.open(image_path)
            w, h = img.size
            
            # Early exit for small images
            if w < 512 or h < 512:
                return crops
            def enc(left, top, right, bottom):
                box = (max(0, int(left)), max(0, int(top)), min(w, int(right)), min(h, int(bottom)))
                if (box[2]-box[0]) < 28 or (box[3]-box[1]) < 28:
                    return None
                c = img.crop(box)
                c.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                if for_cloud:
                    if c.mode != 'RGB':
                        c = c.convert('RGB')
                    c.save(buf, format='JPEG', quality=85, optimize=True)
                else:
                    c.save(buf, format='PNG', optimize=True)
                result = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                return result
            # Center crop: focus on main subject, avoiding extreme edges
            cx0, cy0, cx1, cy1 = 0.28*w, 0.30*h, 0.72*w, 0.70*h
            
            # Extended crop: wider view, but adjust bottom to avoid grass for low-angle shots
            ex0, ey0, ex1, ey1 = 0.15*w, 0.20*h, 0.85*w, 0.75*h
            
            for box in [(cx0, cy0, cx1, cy1), (ex0, ey0, ex1, ey1)]:
                b64 = enc(*box)
                if b64:
                    crops.append(b64)
            
            img.close()
        except Exception as e:
            _log(f"Geometric crops error: {e}")
            return crops
    return crops


def generate_detail_crops(image_path: str, for_cloud: bool = False) -> list[str]:
    """Generate detail crops - uses ONNX Runtime if available, otherwise geometric"""
    # Try ONNX first (intelligent vehicle detection, thread-safe)
    # Generates crops for dominant vehicle only (3 crops: full, left, right)
    try:
        onnx_crops = generate_detail_crops_onnx(image_path, for_cloud)
        if onnx_crops:
            return onnx_crops
    except Exception as e:
        _log(f"ONNX crop generation failed: {e}")
        import traceback
        _log(f"ONNX traceback: {traceback.format_exc()}")
        # Fall through to geometric
    
    # Fallback to geometric crops (stable, no threading issues)
    return generate_detail_crops_geometric(image_path, for_cloud)


def extract_hint_from_path(image_path: str) -> str:
    try:
        p = Path(image_path)
        parts = []
        # Consider parent directories and file stem
        try:
            parent_name = p.parent.name
        except Exception:
            parent_name = ''
        try:
            grandparent_name = p.parent.parent.name if p.parent else ''
        except Exception:
            grandparent_name = ''
        stem = p.stem
        for candidate in (grandparent_name, parent_name, stem):
            if candidate:
                parts.append(candidate)
        raw = ' '.join(parts)
        s = raw.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        tokens = [t for t in s.split() if t]
        filtered = []
        import re as _re
        junk = {'img', 'image', 'photo', 'copy', 'final', 'edited', 'export', 'iphone', 'android', 'dsc', 'hdr', 'raw'}
        for t in tokens:
            tl = t.strip()
            ll = tl.lower()
            if not tl or ll in junk:
                continue
            # Skip long pure numeric tokens and dates
            if ll.isdigit() and len(ll) >= 4:
                continue
            if _re.match(r'^(19|20)\d{2}$', ll):
                continue
            filtered.append(tl)
        hint = ' '.join(filtered[:12]).strip()
        return hint
    except Exception:
        return ''


def classify_vehicle_presence(client, model_name: str, img_b64: str) -> tuple[int, int, bool, float]:
    """Return (cars, motorcycles, is_vehicle, confidence).
    Enhanced multi-vehicle counting for motorsport scenes."""
    try:
        # Enhanced multi-vehicle detection
        _log(f"DEBUG: Using enhanced multi-vehicle classification")
        
        # Use the global _log function instead of defining it locally
        
        # Intelligent prompt for accurate vehicle counting
        prompt = """Analyze this image and count ONLY clearly visible vehicles.
STRICT RULES:
- Count only vehicles that are clearly visible in the foreground or mid‑ground.
- IGNORE distant/background/partially occluded vehicles that occupy a small area (< ~10% of the frame) or are heavily cropped.
- Do NOT count people, logos, reflections, shadows, or inferred vehicles.
- If you cannot clearly see multiple vehicles, prefer Cars: 1 rather than over‑counting.
Examples:
- One main car in foreground; distant cars in the background -> Cars: 1
- Person in racing suit + no car visible -> Cars: 0
Return exactly these lines:
Cars: [number]
Motorcycles: [number]
Total: [number]
Confidence: [0.1-1.0]"""
        
        # Direct chat call with enhanced prompt
        # Increase num_predict to 300 for qwen3-vl to avoid truncation
        response = chat(client, model_name, [{'role': 'user', 'content': prompt, 'images': [img_b64]}], 
                       options_override={"num_predict": 300, "temperature": 0.1})
        
        # DEBUG: Log the raw response before processing
        _log(f"DEBUG: Raw response type: {type(response).__name__}")
        _log(f"DEBUG: Raw response: {repr(response)}")
        _log(f"DEBUG: Raw response length: {len(repr(response))}")
        
        # For qwen3-vl, try content first, then fall back to thinking field
        txt = (extract_message_text(response) or '').strip().lower()
        
        # If content is empty and this is qwen3-vl, try thinking field
        if not txt and isinstance(model_name, str) and ('qwen3' in model_name.lower() or 'qwen-3vl' in model_name.lower()):
            _log(f"DEBUG: Content empty for qwen3-vl classifier, checking thinking field")
            thinking = extract_message_thinking(response)
            if thinking:
                _log(f"DEBUG: Thinking field found, length: {len(thinking)}")
                _log(f"DEBUG: Thinking preview: {thinking[:300]}")
                # Try to extract structured count from thinking
                txt = thinking.strip().lower()
                
                # For thinking field, try to count explicit vehicle mentions
                # Look for patterns like "count as 1", "that's 1", "so that's X", OR numbered lists "1. red ferrari"
                vehicle_count_matches = re.findall(r'count(?:\s+as)?\s+(\d+)|that(?:\'s|s)\s+(\d+)', txt, re.IGNORECASE)
                
                # Also check for numbered vehicle lists like "1. red ferrari" "2. white porsche"
                numbered_list_matches = re.findall(r'^\s*(\d+)\.\s+(?:red|white|black|blue|silver|gray|grey|yellow|green|orange).*?(?:ferrari|porsche|audi|bmw|mercedes|lamborghini|car|vehicle)', txt, re.MULTILINE | re.IGNORECASE)
                
                explicit_count = 0
                if vehicle_count_matches:
                    # Sum up all the counts found from "count" phrases
                    explicit_count = sum(int(m[0] or m[1]) for m in vehicle_count_matches)
                    _log(f"DEBUG: Found explicit vehicle counts in thinking: {explicit_count}")
                elif numbered_list_matches:
                    # Count unique numbers from numbered lists
                    unique_numbers = set(int(n) for n in numbered_list_matches)
                    explicit_count = len(unique_numbers)
                    _log(f"DEBUG: Found numbered vehicle list in thinking: {explicit_count} vehicles")
                
                if explicit_count > 0:
                    # Inject a structured format into txt so the regex can parse it
                    txt = f"cars: {explicit_count}\nmotorcycles: 0\ntotal: {explicit_count}\nconfidence: 0.8\n" + txt
        
        _log(f"Enhanced classifier response: '{txt}'")
        _log(f"Response length: {len(txt)} characters")
        _log(f"Contains 'car': {'car' in txt.lower()}")
        _log(f"Contains 'vehicle': {'vehicle' in txt.lower()}")
        _log(f"Contains numbers: {bool(re.findall(r'\b(\d+)\b', txt))}")
        
        # DEBUG: Confirm we're about to start parsing
        _log(f"DEBUG: About to start parsing section")
        _log(f"DEBUG: Text to parse: '{txt}'")
        
        # Parse the structured response
        cars = 0
        motorcycles = 0
        confidence = 0.5
        
        # DEBUG: Confirm we're starting regex parsing
        _log(f"DEBUG: Starting regex parsing section")
        _log(f"DEBUG: Text to parse: '{txt}'")
        _log(f"DEBUG: About to execute first regex pattern")
        
        # Look for "Cars: X" or "Cars: [X]" pattern (case insensitive)
        _log(f"DEBUG: Looking for cars pattern in text: '{txt}'")
        _log(f"DEBUG: Regex pattern: r'cars?:\\s*\\[?(\\d+)\\]?'")
        car_match = re.search(r'cars?:\s*\[?(\d+)\]?', txt, re.IGNORECASE)
        if car_match:
            cars = int(car_match.group(1))
            _log(f"DEBUG: Found cars match: {car_match.group(0)} -> {cars}")
        else:
            _log(f"DEBUG: No cars match found in text: '{txt}'")
            _log(f"DEBUG: Raw text for debugging: '{repr(txt)}'")
        
        # Look for "Motorcycles: X" or "Motorcycles: [X]" pattern (case insensitive)
        _log(f"DEBUG: Looking for motorcycles pattern in text: '{txt}'")
        moto_match = re.search(r'motorcycles?:\s*\[?(\d+)\]?', txt, re.IGNORECASE)
        if moto_match:
            motorcycles = int(moto_match.group(1))
            _log(f"DEBUG: Found motorcycles match: {moto_match.group(0)} -> {motorcycles}")
        else:
            _log(f"DEBUG: No motorcycles match found in text: '{txt}'")
        
        # Look for "Total: X" only as a weak backup, but do not infer car count blindly
        total_match = re.search(r'total:\s*\[?(\d+)\]?', txt, re.IGNORECASE)
        if total_match and cars == 0 and motorcycles == 0:
            total_vehicles = int(total_match.group(1))
            # If total==1, assume a single car; otherwise leave cars=0 to avoid over-counting
            if total_vehicles == 1:
                cars = 1
        
        # Look for "Confidence: X.X" or "Confidence: [X.X]" pattern
        conf_match = re.search(r'confidence:\s*\[?([0-9.]+)\]?', txt, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # Enhanced fallback: if no structured format found, try intelligent counting
        # BUT only if we haven't already found a valid count from the structured format
        # IMPORTANT: We need to track whether we successfully parsed structured format
        structured_format_parsed = False
        
        # Check if we successfully parsed the structured format
        # We know we parsed it if we have valid numbers and the text contains the expected format
        if cars == 0 and motorcycles == 0 and 'cars: 0' in txt and 'motorcycles: 0' in txt:
            structured_format_parsed = True
            _log(f"DEBUG: Successfully parsed structured format: cars={cars}, motorcycles={motorcycles}")
        
        if not structured_format_parsed and cars == 0 and motorcycles == 0:
            # Only run fallback if we didn't successfully parse structured format
            _log(f"DEBUG: No structured format parsed, running fallback logic")
            # Avoid counting numbers from enumerated lists; only accept explicit labels
            try:
                if 'cars: 1' in txt and 'motorcycles:' in txt:
                    cars = 1
                    confidence = max(confidence, 0.6)
            except Exception as e:
                _log(f"Exception in fallback parsing: {e}")
                _log(f"Text that caused exception: '{txt}'")
        else:
            # We already have valid counts from structured format, don't override them
            _log(f"DEBUG: Skipping fallback parsing - already have cars={cars}, motorcycles={motorcycles}")
            # Skip ALL remaining fallback logic since we have valid counts
            goto_final_result = True
        
        # Skip all remaining fallback logic if we already have valid counts from structured format
        if 'goto_final_result' in locals() and goto_final_result:
            _log(f"DEBUG: Skipping all remaining fallback logic - using structured format results")
        else:
            # Additional fallback: if still no vehicles, check for obvious car indicators
            # BUT only if we haven't already found a valid count from the structured format
            if cars == 0 and motorcycles == 0:
                car_indicators = ['car', 'vehicle', 'ferrari', 'porsche', 'bmw', 'audi', 'mercedes', 'race', 'motorsport']
                if any(indicator in txt.lower() for indicator in car_indicators):
                    cars = 1  # At least one car is present
                    confidence = 0.5
                    _log(f"DEBUG: Fallback detection: car indicators found in text: '{txt}'")
            
            # REMOVED: TEMPORARY FIX that was forcing car detection incorrectly
            
            # ULTIMATE FALLBACK: If we see "cars: 1" in the response but regex failed, force it
            # BUT only if we haven't already found a valid count from the structured format
            if cars == 0 and motorcycles == 0 and 'cars: 1' in txt:
                cars = 1
                confidence = 0.8
                _log(f"ULTIMATE FALLBACK: Forcing cars=1 because 'cars: 1' found in response: '{txt}'")
            elif cars == 0 and motorcycles == 0 and 'cars: 0' in txt:
                # If we explicitly see "cars: 0", respect that and don't force detection
                _log(f"DEBUG: Respecting explicit 'cars: 0' from classifier response")
                confidence = 0.9  # High confidence in the explicit zero count
        
        # Cap maximum count for sanity
        cars = min(cars, 20)
        motorcycles = min(motorcycles, 10)
        
        is_vehicle = (cars + motorcycles) > 0
        
        _log(f"Final result: cars={cars}, motorcycles={motorcycles}, is_vehicle={is_vehicle}, confidence={confidence}")
        
        # DEBUG: Confirm we completed regex parsing
        _log(f"DEBUG: Completed regex parsing section")
        _log(f"DEBUG: Final values: cars={cars}, motorcycles={motorcycles}, confidence={confidence}")
        _log(f"DEBUG: About to return: cars={cars}, motorcycles={motorcycles}, is_vehicle={is_vehicle}, confidence={confidence}")
        
        return cars, motorcycles, is_vehicle, confidence
            
    except Exception as e:
        print(f"Simple classification failed: {e}")
        return 0, 0, False, 0.3


def is_single_dominant_vehicle(client, model_name: str, img_b64: str) -> bool:
    """Ultra-fast check to avoid accidental motorshow routing.

    Returns True if there is a single, clearly dominant vehicle centered in the frame.
    """
    try:
        prompt = (
            "Answer ONLY 'Yes' or 'No'. Is there exactly one clearly dominant vehicle centered in the image?"
        )
        resp = chat(
            client,
            model_name,
            [{'role': 'user', 'content': prompt, 'images': [img_b64]}],
            options_override={"num_predict": 4, "temperature": 0.0},
        )
        txt = (extract_message_text(resp) or '').strip().lower()
        return txt.startswith('y')  # yes
    except Exception:
        return False

def classify_is_car(client, model_name: str, img_b64: str) -> bool:
    """Backwards-compatible: True if at least one car/motorcycle is present with minimal confidence."""
    try:
        cars, motos, isv, conf = classify_vehicle_presence(client, model_name, img_b64)
        if (cars + motos) >= 1 and (conf >= 0.40 or cars >= 1):
            return True
        return False
    except Exception:
        return False


def estimate_vehicle_count(client, model_name: str, img_b64: str) -> int:
    """Return total of cars + motorcycles using the unified classifier."""
    try:
        cars, motos, _isv, _conf = classify_vehicle_presence(client, model_name, img_b64)
        return max(0, min(5, int(cars) + int(motos)))
    except Exception:
        return 0


def scene_description_220(client, model_name: str, img_b64: str) -> str:
    """Generate a richer non-car scene description, capped at ~220 words."""
    try:
        # Use the new prompt template system
        context = {'use_cloud': False}  # Local model for richer descriptions
        prompt = get_prompt('non_car_analysis', context)
        
        # Track performance
        start_time = time.time()
        
        # Use smart retry with fallback
        response = smart_retry_with_fallback(client, model_name, prompt, img_b64, max_retries=1)
        
        # Record metrics
        response_time = time.time() - start_time
        success = bool(response and len(response.strip()) > 50)
        prompt_metrics.record_prompt_result('non_car_analysis', success, response_time)
        
        if not success:
            prompt_metrics.record_fallback_usage('non_car_analysis')
            # Fallback to original prompt if new system fails
            fallback_prompt = (
                'Write a vivid, highly descriptive yet factual scene description (120–220 words). '
                'Use precise, sensory language. Describe: foreground/midground/background; composition and perspective; horizon placement; reflections; '
                'dominant color palette and gradients; light quality (soft/harsh, warm/cool), shadows and highlights; textures and materials; weather/atmosphere; '
                'mood and ambience. Quote any readable signage exactly. Avoid apologies, instructions, or speculation beyond what is visible. '
                'Do not mention cars unless clearly present. No lists—write as flowing prose. '
                'IMPORTANT: Avoid repeating the same words or concepts. Use varied vocabulary and descriptions.'
            )
            resp = chat(client, model_name, [{'role': 'user', 'content': fallback_prompt, 'images': [img_b64]}], options_override={"num_predict": 260})
            response = extract_message_text(resp) or ''
        
        # Process response
        desc = response or ''
        words = desc.split()
        if len(words) > 220:
            desc = ' '.join(words[:220])
        return desc
    except Exception as e:
        print(f"Scene description failed: {e}")
        prompt_metrics.record_fallback_usage('non_car_analysis')
        return ''


def car_confidence_from_fields(parsed: dict, primary_text: str) -> float:
    """Heuristic car confidence from parsed fields and text cues (0..1)."""
    try:
        score = 0.0
        if parsed.get('Make') and str(parsed['Make']).strip().lower() not in {'unknown', ''}:
            score += 0.35
        if parsed.get('Model') and str(parsed['Model']).strip().lower() not in {'unknown', ''}:
            score += 0.35
        if parsed.get('License Plate') and str(parsed['License Plate']).strip().lower() not in {'unknown', ''}:
            score += 0.2
        if parsed.get('Logos') and str(parsed['Logos']).strip():
            score += 0.1
        # Race numbers are strong motorsport evidence of a car subject
        if parsed.get('Race Number') and str(parsed['Race Number']).strip().lower() not in {'unknown', ''}:
            score += 0.12
        # Text cues
        p_score, c_score = person_vs_car_score(primary_text)
        if c_score > 0:
            score += min(0.2, 0.05 * c_score)
        if p_score > c_score:
            score -= min(0.3, 0.05 * (p_score - c_score))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def summary_from_text_220(text: str) -> str:
    """Extract a concise, neutral summary from free text; cap ~220 words and strip meta/apology lines."""
    try:
        import re
        s = (text or '').strip()
        # Prefer explicit key if present
        m = re.search(r"AI[-\s]?Interpretation\s+Summary\s*:\s*(.+)", s, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
        else:
            # First meaningful sentence(s)
            parts = re.split(r"(?<=[.!?])\s+", s)
            candidate = ''
            for p in parts:
                pl = p.lower()
                if not p.strip():
                    continue
                # Skip meta/apologies
                if any(tok in pl for tok in ['apolog', 'cannot', "can't", 'not possible', 'as an ai', 'misunderstanding']):
                    continue
                candidate = p.strip()
                break
        words = candidate.split()
        if len(words) > 220:
            candidate = ' '.join(words[:220])
        return candidate
    except Exception:
        return ''


def person_vs_car_score(text: str) -> tuple[int, int]:
    """Heuristic scoring from free text: returns (person_score, car_score)."""
    try:
        low = (text or '').lower()
        person_tokens = [
            'person', 'people', 'woman', 'man', 'girl', 'boy', 'model ', 'models ', 'presenter', 'dressed',
            'portrait', 'smiling', 'standing', 'posing', 'crowd', 'audience', 'booth girl', 'hostess',
            # Motorshow/exhibition cues
            'booth', 'stand', 'stage', 'exhibition', 'expo', 'motorshow', 'auto show', 'promotional staff',
            'spokesmodel', 'concept car', 'prototype', 'display', 'show floor', 'press event', 'launch',
            'presentation', 'podium'
        ]
        car_tokens = [
            'car', 'vehicle', 'automobile', 'grille', 'tail light', 'taillight', 'headlight', 'bumper', 'hood',
            'fender', 'door handle', 'windshield', 'exhaust', 'diffuser', 'badge', 'logo on the grille',
            # racing/show cues
            'race car', 'racing', 'livery', 'spoiler', 'wing', 'gt', 'gt3', '911', 'supercar', 'sports car',
            # common brand cues that often appear in captions
            'porsche', 'ferrari', 'lamborghini', 'bmw', 'audi', 'mercedes', 'ford', 'chevrolet', 'toyota',
            'mobil', 'michelin', 'pirelli', 'sponsor', 'decal'
        ]
        p = sum(low.count(tok) for tok in person_tokens)
        c = sum(low.count(tok) for tok in car_tokens)
        return p, c
    except Exception:
        return (0, 0)


def _find_exiftool() -> str | None:
    """Locate exiftool across common locations (supports PyInstaller onefile, cross-platform).
    On Windows: looks for exiftool.exe
    On Mac/Linux: looks for exiftool in system PATH and common locations.
    """
    try:
        import sys as _sys
        import shutil
        import platform
        
        candidates: list[str] = []
        is_windows = platform.system() == 'Windows' or os.name == 'nt'
        exiftool_name = 'exiftool.exe' if is_windows else 'exiftool'
        
        # On Mac/Linux, check system PATH first (Homebrew installs it there)
        if not is_windows:
            exiftool_path = shutil.which('exiftool')
            if exiftool_path:
                candidates.append(exiftool_path)
            # Also check common Homebrew locations
            candidates.extend([
                '/usr/local/bin/exiftool',
                '/opt/homebrew/bin/exiftool',
                '/usr/bin/exiftool',
            ])
        
        # PyInstaller temp dir
        try:
            meipass = getattr(_sys, '_MEIPASS', '')
            if meipass:
                candidates.append(os.path.join(meipass, exiftool_name))
                if is_windows:
                    candidates.append(os.path.join(meipass, 'exiftool_files', exiftool_name))
        except Exception:
            pass
        
        # Executable directory
        try:
            if getattr(_sys, 'frozen', False) and getattr(_sys, 'executable', ''):
                exe_dir = os.path.dirname(_sys.executable)
                candidates.append(os.path.join(exe_dir, exiftool_name))
                if is_windows:
                    candidates.append(os.path.join(exe_dir, 'exiftool_files', exiftool_name))
        except Exception:
            pass
        
        # Current working dir and script dir
        try:
            candidates.extend([
                os.path.join(os.getcwd(), exiftool_name),
                os.path.join(os.path.dirname(__file__), exiftool_name),
                exiftool_name,
                f'./{exiftool_name}',
            ])
        except Exception:
            candidates.extend([exiftool_name, f'./{exiftool_name}'])
        
        return next((p for p in candidates if p and os.path.exists(p)), None)
    except Exception:
        return None


def _exiftool_run_kwargs(exe_path: str, timeout: int = 30) -> dict:
    """Build subprocess.run kwargs so ExifTool works from PyInstaller onefile (cross-platform).
    - Windows: Sets PERL5LIB/PATH to locate perl5*.dll and modules when using perl-wrapper
    - Windows: Hides console window
    - Mac/Linux: Uses system ExifTool, minimal environment setup
    - Sets cwd to exiftool folder (if bundled) or current dir (if system)
    """
    import subprocess as _subprocess
    import platform
    
    is_windows = platform.system() == 'Windows' or os.name == 'nt'
    env = os.environ.copy()
    
    # Determine working directory
    # For bundled exiftool (Windows), use its directory for Perl libs
    # For system exiftool (Mac/Linux), use current directory
    exe_dir = os.path.dirname(os.path.abspath(exe_path)) if os.path.dirname(exe_path) else None
    if exe_dir and os.path.isdir(exe_dir):
        base_dir = exe_dir
    else:
        base_dir = os.getcwd()
    
    # Windows-specific: Perl wrapper support
    if is_windows and exe_dir:
        perl_lib = os.path.join(exe_dir, 'exiftool_files', 'lib')
        if os.path.isdir(perl_lib):
            env['PERL5LIB'] = perl_lib
            env['PATH'] = f"{os.path.join(exe_dir, 'exiftool_files')};{env.get('PATH','')}"
    
    kwargs = {"capture_output": True, "text": True, "timeout": timeout, "cwd": base_dir, "env": env}
    
    # Windows-specific: Hide console window
    try:
        if is_windows:
            si = _subprocess.STARTUPINFO()
            si.dwFlags |= _subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = getattr(_subprocess, 'CREATE_NO_WINDOW', 0)
    except Exception:
        pass
    
    return kwargs


def chat(client, model_name: str, messages: list[dict], options_override: dict | None = None, motorshow_mode: bool = False):
    """Unified chat router: prefer cloud if configured, else local Ollama."""
    opts = {
        "temperature": 0.2,
        "top_p": 0.8,
        "num_predict": 192,
    }
    if isinstance(options_override, dict):
        opts.update(options_override)
    
    # Qwen-3VL: /no_think directive is unreliable for complex prompts, skip it
    # if isinstance(model_name, str) and ('qwen3' in model_name.lower() or 'qwen-3vl' in model_name.lower()):
    #     for msg in messages:
    #         if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
    #             if not msg['content'].startswith('/no_think'):
    #                 msg['content'] = '/no_think\n' + msg['content']
    #             break
    pass
    
    # If cloud/external preferred, always use router (no local fallback)
    try:
        # Trust persisted settings first (in case router state didn't update)
        s = QSettings('Pistonspy', 'TagOmatic')
        settings_use_cloud = bool(s.value('cloud/use_cloud', False, type=bool))
        settings_provider = s.value('cloud/preferred_provider', 'auto', type=str)
    except Exception:
        settings_use_cloud = False
        settings_provider = ''
    
    if settings_use_cloud and settings_provider in {'remote-ollama', 'openai-compatible', 'auto'}:
        try:
            # Direct-call for providers that don't require SDKs
            if settings_provider == 'openai-compatible':
                try:
                    from urllib import request as _urlreq
                    import json as _json
                    base = s.value('cloud/oai_compat_base_url', 'http://localhost:1234/v1', type=str) or 'http://localhost:1234/v1'
                    api_key = s.value('cloud/oai_compat_api_key', '', type=str) or ''
                    # Prefer saved model; ignore placeholder or local dropdown text
                    model = s.value('cloud/oai_compat_model', '', type=str)
                    if not model or model.strip() == '' or str(model).strip().lower() in {'(none - cloud)','none','null'}:
                        try:
                            model = getattr(cloud_router, 'oai_compat_model', '')
                        except Exception:
                            model = ''
                    if not model:
                        raise ValueError('OpenAI-compatible model not set. Choose a model in Cloud Backends and Save.')
                    url = (base.rstrip('/') + '/chat/completions')
                    oai_msgs = []
                    for m in messages:
                        # Build OpenAI content array with optional data-URI images
                        if m.get('images'):
                            content = []
                            if m.get('content'):
                                content.append({"type": "text", "text": str(m.get('content'))})
                            for b64 in (m.get('images') or []):
                                if isinstance(b64, str) and len(b64) > 10:
                                    data_uri = 'data:image/jpeg;base64,' + b64
                                    content.append({"type": "image_url", "image_url": {"url": data_uri}})
                            oai_msgs.append({"role": m.get('role','user'), "content": content or str(m.get('content',''))})
                        else:
                            oai_msgs.append({"role": m.get('role','user'), "content": str(m.get('content',''))})
                    body = {
                        "model": model,
                        "messages": oai_msgs,
                        "stream": False,
                        # Map simple generation options if supported
                        "temperature": float(opts.get('temperature', 0.2)),
                        "top_p": float(opts.get('top_p', 0.8))
                    }
                    data = _json.dumps(body).encode('utf-8')
                    headers = {"Content-Type": "application/json"}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    req = _urlreq.Request(url, data=data, headers=headers, method='POST')
                    resp = _urlreq.urlopen(req, timeout=20)
                    raw = resp.read().decode('utf-8', 'ignore')
                    obj = _json.loads(raw)
                    text = ''
                    try:
                        text = str((((obj.get('choices') or [{}])[0]).get('message') or {}).get('content',''))
                    except Exception:
                        text = ''
                    return {"_provider": "openai-compatible", "message": {"role": "assistant", "content": text}}
                except Exception as _e_oai:
                    # Fallback to router if direct call fails
                    pass
            if settings_provider == 'external-ollama':
                try:
                    from urllib import request as _urlreq
                    import json as _json
                    host = s.value('cloud/ollama_host', 'http://localhost:11434', type=str) or 'http://localhost:11434'
                    model = s.value('cloud/ollama_model', '', type=str) or model_name
                    url = host.rstrip('/') + '/api/chat'
                    payload_msgs = []
                    for m in messages:
                        payload_msgs.append({
                            'role': m.get('role','user'),
                            'content': str(m.get('content','')),
                            'images': m.get('images') or None
                        })
                    body = {
                        'model': model,
                        'messages': payload_msgs,
                        'options': {
                            'temperature': float(opts.get('temperature', 0.2)),
                            'top_p': float(opts.get('top_p', 0.8)),
                            'num_predict': int(opts.get('num_predict', 192)),
                        }
                    }
                    data = _json.dumps(body).encode('utf-8')
                    headers = {"Content-Type": "application/json"}
                    req = _urlreq.Request(url, data=data, headers=headers, method='POST')
                    resp = _urlreq.urlopen(req, timeout=20)
                    raw = resp.read().decode('utf-8', 'ignore')
                    obj = _json.loads(raw)
                    # Ollama /api/chat returns {'message': {'content': '...'}}
                    return {"_provider": "external-ollama", "message": (obj.get('message') or {"role":"assistant","content": ""})}
                except Exception as _e_ol:
                    pass
            # Default to router for OpenAI/Google/auto
            return cloud_router.chat(messages=messages, images_b64=None, options_override=opts)
        except Exception:
            pass

    if getattr(cloud_router, 'use_cloud', False):
        try:
            prov = getattr(cloud_router, 'preferred_provider', '') or ''
            if prov in {'remote-ollama', 'openai-compatible', 'auto'}:
                return cloud_router.chat(messages=messages, images_b64=None, options_override=opts)
        except Exception:
            pass
    # Otherwise use local (final fallback - always executes if cloud paths didn't return)
    try:
        return client.chat(model=model_name, messages=messages, keep_alive="30m", options=opts)
    except Exception as e:
        _log(f"Local ollama chat failed: {e}")
        # Return minimal valid response instead of None
        return {"message": {"content": ""}}
def noncar_describe_5(client, model_name: str, img_b64: str, motorshow_mode: bool = False) -> dict:
    """2-Phase approach: Generate concise prose first, then extract structured fields.
    Phase 1: Brief description (75-125 words, ~150-300 chars) - automotive journalism style if motorshow_mode
    Phase 2: Parse into structured NC_* fields"""
    
    # Validate input image data
    if not img_b64 or not isinstance(img_b64, str) or len(img_b64.strip()) < 100:
        print(f"[DEBUG] noncar_describe_5: Invalid image data - length: {len(img_b64) if img_b64 else 0}, type: {type(img_b64)}")
        return {
            'Image Type': 'Photograph',
            'Primary Subject': 'Image Content',
            'Setting': 'Image Environment', 
            'Key Objects': '',
            'Descriptive Summary': 'Image analysis failed - invalid data'
        }
    
    try:
        # Phase 1: Generate concise, focused summary (75-125 words)
        if motorshow_mode:
            # Automotive journalism style for motorshow/automotive events
            phase1_prompt = (
                'Describe this automotive/motorsport scene in vivid, journalistic prose (100-200 words). '
                'Write like an automotive journalist capturing the atmosphere of a car show, race event, or automotive gathering. '
                'Focus on: the vehicles visible, the event atmosphere, crowd/spectators, venue details, lighting, and overall ambiance. '
                'Mention any visible brands, liveries, or automotive details. '
                'Be descriptive and evocative - paint a picture of the automotive scene and its energy. '
                'If no cars are clearly visible, describe the automotive environment and context. '
                'Provide ONLY the description, no thinking or reasoning.'
            )
        else:
            # Standard descriptive style for general scenes
            # For qwen3-vl: explicitly request content-only output
            phase1_prompt = (
                'Describe this image in rich, descriptive prose (100-200 words). '
                'Focus on the main subject, key objects, setting, atmosphere, colors, textures, and details. '
                'Be descriptive and vivid - paint a picture with words. '
                'Write naturally about what you see, emphasizing the most important and interesting elements. '
                'Provide ONLY the description, no thinking or reasoning.'
            )
        print(f"[DEBUG] Phase 1 Prompt: {phase1_prompt}")
        print(f"[DEBUG] Phase 1 Model: {model_name}")
        print(f"[DEBUG] Phase 1 Image Length: {len(img_b64) if img_b64 else 0}")
        try:
            # Increased tokens to avoid truncation (probe showed 287 chars got cut off at 250 tokens)
            num_predict_tokens = 400 if motorshow_mode else 350
            phase1_resp = chat(client, model_name, [{'role': 'user', 'content': phase1_prompt, 'images': [img_b64]}], options_override={"num_predict": num_predict_tokens})
            print(f"[DEBUG] Phase 1 Raw Response: {phase1_resp}")
            rich_prose = extract_message_text(phase1_resp) or ''
            
            # If qwen3-vl returns empty content or content is too short, check thinking field
            if (not rich_prose or len(rich_prose.strip()) < 100) and isinstance(model_name, str) and ('qwen3' in model_name.lower() or 'qwen-3vl' in model_name.lower()):
                print(f"[DEBUG] Phase 1 content insufficient for qwen3-vl ({len(rich_prose)} chars), trying simpler prompt")
                # Try ultra-simple direct question
                simple_prompt = "Describe this image in detail (150-200 words). Provide ONLY your description, no thinking or reasoning."
                try:
                    simple_resp = chat(client, model_name, [{'role': 'user', 'content': simple_prompt, 'images': [img_b64]}], options_override={"num_predict": 400, "temperature": 0.4})
                    simple_content = extract_message_text(simple_resp) or ''
                    if simple_content and len(simple_content.strip()) > len(rich_prose.strip()):
                        rich_prose = simple_content
                        print(f"[DEBUG] Simple prompt succeeded: {len(rich_prose)} chars")
                except Exception as simple_error:
                    print(f"[DEBUG] Simple prompt failed: {simple_error}")
                    pass
        except Exception as chat_error:
            print(f"[DEBUG] Phase 1 chat error: {type(chat_error).__name__}: {chat_error}")
            rich_prose = ''
        print(f"[DEBUG] Phase 1 Response Length: {len(rich_prose)}")
        print(f"[DEBUG] Phase 1 Response Preview: {rich_prose[:200]}...")
        
        # If Phase 1 fails, retry with a more specific prompt
        if not rich_prose or len(rich_prose.strip()) < 30:
            retry_prompt = (
                'Look at this image carefully and describe what you see in clear detail (50-100 words). '
                'Focus on the main subject, key objects, setting, and atmosphere. '
                'Be descriptive but concise about the scene and environment. '
                'Describe as if to someone who cannot see the image. Do not mention cars unless present.'
            )
            try:
                retry_resp = chat(client, model_name, [{'role': 'user', 'content': retry_prompt, 'images': [img_b64]}], options_override={"num_predict": 120})
                rich_prose = extract_message_text(retry_resp) or ''
            except Exception as retry_error:
                print(f"[DEBUG] Phase 1 retry chat error: {type(retry_error).__name__}: {retry_error}")
                rich_prose = ''
        
        # Phase 2: For qwen3-vl, skip LLM extraction (unreliable for text-only) and parse directly from Phase 1
        if rich_prose and len(rich_prose.strip()) >= 20:
            print(f"[DEBUG] Phase 2: Parsing Phase 1 description directly (qwen3-vl text-only unreliable)")
            
            # Direct parsing from rich_prose - extract key elements
            out = {
                'Image Type': 'Photograph',  # Default assumption
                'Primary Subject': '',
                'Setting': '',
                'Key Objects': '',
                'Descriptive Summary': rich_prose[:500]  # Use full Phase 1 description
            }
            
            # Simple keyword extraction for Primary Subject
            prose_lower = rich_prose.lower()
            if 'crown' in prose_lower:
                out['Primary Subject'] = 'Crown'
            elif 'car' in prose_lower or 'vehicle' in prose_lower:
                out['Primary Subject'] = 'Vehicle'
            elif 'person' in prose_lower or 'people' in prose_lower:
                out['Primary Subject'] = 'Person'
            elif 'building' in prose_lower or 'architecture' in prose_lower:
                out['Primary Subject'] = 'Architecture'
            elif 'landscape' in prose_lower or 'scenery' in prose_lower:
                out['Primary Subject'] = 'Landscape'
            else:
                # Extract first noun from description
                words = rich_prose.split()
                for word in words[:20]:  # Check first 20 words
                    if len(word) > 3 and word[0].isupper():
                        out['Primary Subject'] = word.strip('.,!?')
                        break
            
            # Extract setting if mentioned
            if 'museum' in prose_lower:
                out['Setting'] = 'Museum'
            elif 'outdoor' in prose_lower or 'outside' in prose_lower:
                out['Setting'] = 'Outdoor'
            elif 'indoor' in prose_lower or 'inside' in prose_lower:
                out['Setting'] = 'Indoor'
            elif 'backdrop' in prose_lower or 'background' in prose_lower:
                # Extract what follows "backdrop" or "background"
                for phrase in ['backdrop', 'background']:
                    if phrase in prose_lower:
                        idx = prose_lower.find(phrase)
                        snippet = rich_prose[idx:idx+50]
                        out['Setting'] = snippet.split('.')[0].strip()
                        break
            
            # Extract key objects - look for prominent nouns
            key_terms = []
            object_keywords = ['diamond', 'gold', 'silver', 'emerald', 'ruby', 'pearl', 'cross', 'eagle', 'velvet', 'jewel', 'stone']
            for keyword in object_keywords:
                if keyword in prose_lower:
                    key_terms.append(keyword.capitalize())
            out['Key Objects'] = ', '.join(key_terms[:5]) if key_terms else ''
            
            print(f"[DEBUG] Final Parsed Output (direct): {out}")
            return out
        else:
            print(f"[DEBUG] Phase 1 failed to produce usable prose, using fallback")
            return fallback_scene_analysis()
            
    except Exception as e:
        print(f"[DEBUG] Exception in noncar_describe_5: {type(e).__name__}: {e}")
        return fallback_scene_analysis()

def fallback_scene_analysis() -> dict:
    """Simple fallback for when structured analysis fails"""
    return {
        'Image Type': 'Photograph',
        'Primary Subject': 'Scene',
        'Setting': 'Environment',
        'Key Objects': 'Various objects',
        'Descriptive Summary': 'A photograph of a scene or environment'
    }

def infer_enhanced(client, model_name: str, image_path: str, feedback_context: dict | None = None, motorshow_mode: bool = False, noncar_model: str | None = None, motorshow_disabled: bool = False, race_mode: bool = False) -> tuple[str, dict]:
    """
    Enhanced inference that completely ignores existing metadata during processing.
    Each image is treated as fresh for clean, independent analysis.
    """
    try:
        # Prefer cloud-safe JPEG scaling to reduce latency and bandwidth
        use_cloud = False
        try:
            use_cloud = bool(getattr(cloud_router, 'use_cloud', False) and (cloud_router.openai_compat.is_configured() or cloud_router.remote_ollama.is_configured()))
        except Exception:
            use_cloud = False
        if use_cloud:
            # Use simple, fast PIL operations like the working backup
            img_b64 = encode_image_jpeg_b64(image_path, max_size=2048, quality=85)
        else:
            # Use simple, fast PIL operations like the working backup
            img_b64 = encode_image_jpeg_b64(image_path, max_size=2048, quality=85)
    except Exception as e:
        print(f"[DEBUG] Image encoding fallback for {image_path}: {type(e).__name__}: {e}")
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Optional weak hint derived from file path
    hint = extract_hint_from_path(image_path)
    weak_hint_text = f"\nFile-path hint (use only as a weak prior; ignore if inconsistent with the image): {hint}\n" if hint else ""

    # Use the logging function to capture debug output (define first before using)
    def _log(msg: str) -> None:
        try:
            with open(os.path.join(os.getcwd(), 'log.log'), 'a', encoding='utf-8') as lf:
                lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] {msg}\n")
        except Exception:
            pass
    
    # Early routing: classify first; if non-car, directly generate scene with the selected non-car model
    # Use the same model for classification to minimize VRAM usage (important for <4090 GPUs)
    classifier_model = model_name
    _log(f"Using {classifier_model} for classification (same as analysis model to minimize VRAM)")
    
    _log(f"About to call classifier with model: {classifier_model}")
    try:
        # Single classifier call that gives us both car status and vehicle count
        _log("Calling classify_vehicle_presence...")
        _log(f"Input image_b64 length: {len(img_b64) if img_b64 else 'None'}")
        _log(f"Input client type: {type(client)}")
        _log(f"Input classifier_model: {classifier_model}")
        
        cars, motos, is_vehicle, conf = classify_vehicle_presence(client, classifier_model, img_b64)
        
        _log(f"Classifier returned: cars={cars}, motos={motos}, is_vehicle={is_vehicle}, conf={conf}")
        _log(f"Classifier return types: cars={type(cars)}, motos={type(motos)}, is_vehicle={type(is_vehicle)}, conf={type(conf)}")
        
        cls_is_car = is_vehicle
        vc = cars + motos
        
        _log(f"After assignment: cls_is_car={cls_is_car} (type: {type(cls_is_car)}), vc={vc} (type: {type(vc)})")
        _log(f"Routing Decision: cls_is_car={cls_is_car}, vc={vc}")
        
    except Exception as e:
        _log(f"Classifier Exception: {e}")
        _log(f"Exception type: {type(e).__name__}")
        import traceback
        _log(f"Traceback: {traceback.format_exc()}")
        cls_is_car = None
        vc = 0  # Don't force car mode on classifier failure - default to scene
        _log(f"Exception handler set: cls_is_car={cls_is_car}, vc={vc}")
    # If filename/path strongly indicates a car (e.g., brand/model tokens), do NOT early-route to non-car
    hint_l = (hint or '').lower()
    car_tokens = ['porsche','ferrari','lamborghini','bmw','audi','mercedes','amg','ford','chevrolet','toyota','nissan','mazda','honda','subaru','gt3','911','gt','rs','m3','m4','m5','f40','f50','458','488','r8','gtr','supra','rx7','s2000','civic','mustang','camaro']
    hint_screams_car = any(t in hint_l for t in car_tokens)
    # Route decision debug snapshot
    route_debug = {
        'classifier_model': classifier_model,
        'cls_is_car': bool(cls_is_car) if isinstance(cls_is_car, bool) else None,
        'vehicle_count': int(vc) if isinstance(vc, (int, float, str)) else None,
        'hint': hint or '',
        'hint_screams_car': bool(hint_screams_car),
    }

    print(f"[DEBUG] Routing Check: cls_is_car={cls_is_car} (False?), vc={vc} (0?), hint_screams_car={hint_screams_car}")
    print(f"[DEBUG] About to check routing condition...")
    
    # Disable motorshow mode for qwen3 models (wildcard match)
    is_qwen3_model = isinstance(model_name, str) and 'qwen3' in model_name.lower()
    if is_qwen3_model:
        motorshow_mode = False
        _log(f"Motorshow mode disabled for qwen3 model: {model_name}")
    
    # MULTI-VEHICLE MOTORSHOW TRIGGER: Auto-switch to motorshow mode for multi-vehicle scenes
    # Trigger for 2+ cars to get "Journalist Petrol Head" style descriptions
    # Skip if qwen3 model (motorshow already disabled above)
    if vc >= 2 and not motorshow_mode and not is_qwen3_model:
        _log(f"Checking if single dominant vehicle (vc={vc})")
        is_dominant = is_single_dominant_vehicle(client, classifier_model, img_b64)
        _log(f"Single dominant vehicle check: {is_dominant}")
        if not is_dominant:
            print(f"[DEBUG] MULTI-VEHICLE DETECTED ({vc} vehicles) - AUTO-SWITCHING TO MOTORSHOW MODE")
            _log(f"MULTI-VEHICLE DETECTED ({vc} vehicles) - AUTO-SWITCHING TO MOTORSHOW MODE")
            motorshow_mode = True
            print(f"[DEBUG] Motorshow mode auto-enabled for multi-vehicle scene")
        else:
            _log(f"Single dominant vehicle detected, skipping motorshow mode")
    
    # Additional motorshow trigger for motorsport scenes (keep original threshold)
    # Skip if qwen3 model (motorshow already disabled above)
    if hint_screams_car and vc >= 3 and not motorshow_mode and not is_qwen3_model and not is_single_dominant_vehicle(client, classifier_model, img_b64):
        print(f"[DEBUG] MOTORSPORT SCENE DETECTED (hint: {hint}) - AUTO-SWITCHING TO MOTORSHOW MODE")
        motorshow_mode = True
        print(f"[DEBUG] Motorshow mode auto-enabled for motorsport scene")
    
    # ENABLE NON-CAR ROUTING: Route to non-car path when classifier says no cars
    _log(f"Checking routing condition: cls_is_car={cls_is_car}, vc={vc}")
    if cls_is_car is False and vc == 0:
        _log("ROUTING TO NON-CAR PATH (classifier decision)")
        scene_model = noncar_model if noncar_model else model_name
        # Prefer structured 5-line non-car description
        try:
            nc = noncar_describe_5(client, scene_model, img_b64, motorshow_mode=motorshow_mode)
        except Exception:
            nc = {'Image Type': 'N/A', 'Primary Subject': 'N/A', 'Setting': 'N/A', 'Key Objects': '', 'Descriptive Summary': ''}
        # Clean, efficient 2-phase approach: Use the NC fields from noncar_describe_5
        desc_nc = (nc.get('Descriptive Summary') or '').strip()
        
        # If the 2-phase approach didn't produce good results, enhance what we have
        if not desc_nc or len(desc_nc.strip()) < 30:
            # Smart fallback: derive meaningful content from available NC fields
            try:
                primary = nc.get('Primary Subject', '')
                setting = nc.get('Setting', '')
                
                if primary and primary != 'N/A' and setting and setting != 'N/A':
                    desc_nc = f"{primary} in {setting}"
                elif primary and primary != 'N/A':
                    desc_nc = primary
                elif setting and setting != 'N/A':
                    desc_nc = setting
                else:
                    desc_nc = "Scene description"
            except Exception:
                desc_nc = "Scene description"
        # Clean, focused non-car metadata - no empty car fields
        parsed_early = {
            'NC_ImageType': nc.get('Image Type', 'Photograph'),
            'NC_Primary': nc.get('Primary Subject', 'Natural Scene'),
            'NC_Setting': nc.get('Setting', 'Outdoor Environment'),
            'NC_KeyObjects': nc.get('Key Objects', ''),
            'TagoMatic AI Summary': (desc_nc or 'Natural outdoor scene'),
            'LLM Backend': f"Ollama: {scene_model}",
            'Image Type': 'Non-Car Scene'
        }
        # Simplified backend info - no complex fallback logic needed
        return (desc_nc or ''), parsed_early

    _log("ROUTING TO CAR PATH")
    print(f"[DEBUG] ROUTING TO CAR PATH")
    print(f"[DEBUG] About to get prompt template...")
    
    # MULTI-VEHICLE SCENE DESCRIPTION: If multiple vehicles detected, generate scene description first
    if vc >= 2 and motorshow_mode:
        print(f"[DEBUG] MULTI-VEHICLE SCENE: Generating motorsport scene description for {vc} vehicles")
        _log(f"MULTI-VEHICLE SCENE: Generating motorsport scene description for {vc} vehicles")
        try:
            # Generate crops for multi-vehicle identification (same as single-car path)
            multi_vehicle_images = [img_b64]
            if not use_cloud:
                try:
                    _log(f"Generating detail crops for multi-vehicle scene: {image_path}")
                    crops = generate_detail_crops(image_path, for_cloud=False)
                    if crops:
                        multi_vehicle_images.extend(crops[:3])  # 3 crops for dominant vehicle
                        _log(f"Multi-vehicle: Generated {len(crops[:3])} detail crops")
                    else:
                        _log(f"Multi-vehicle: No detail crops generated, using main image only")
                except Exception as crop_error:
                    _log(f"Multi-vehicle crop generation failed: {crop_error}")
                    # Continue with main image only if crops fail
            
            # Step 1: Identify each visible vehicle (Make/Model) - optimized based on probe results
            vehicle_list = []
            # Use simpler, more direct prompt for cloud providers (LLMStudio, Remote Ollama, etc.)
            if use_cloud:
                identify_prompt = (
                    f"Identify the vehicles in this image. For each clearly visible car, provide:\n\n"
                    f"Vehicle 1:\n"
                    f"Make: <brand>\n"
                    f"Model: <model>\n"
                    f"Color: <color>\n"
                    f"RaceNumber: <number or None>\n"
                    f"LicensePlate: <plate or None>\n\n"
                    f"Vehicle 2:\n"
                    f"Make: <brand>\n"
                    f"Model: <model>\n"
                    f"Color: <color>\n"
                    f"RaceNumber: <number or None>\n"
                    f"LicensePlate: <plate or None>\n\n"
                    f"Continue for up to {min(vc, 4)} vehicles. Use exact model names. Provide ONLY the vehicle details in this format."
                )
            else:
                identify_prompt = (
                    f"List the vehicles visible in this image. For each clearly visible car, provide:\n"
                    f"Vehicle 1:\n"
                    f"Make: <brand>\n"
                    f"Model: <model>\n"
                    f"Color: <color>\n"
                    f"RaceNumber: <number if visible, otherwise 'None'>\n"
                    f"LicensePlate: <plate if visible, otherwise 'None'>\n\n"
                    f"Vehicle 2:\n"
                    f"Make: <brand>\n"
                    f"Model: <model>\n"
                    f"Color: <color>\n"
                    f"RaceNumber: <number if visible, otherwise 'None'>\n"
                    f"LicensePlate: <plate if visible, otherwise 'None'>\n\n"
                    f"Continue for up to {min(vc, 4)} vehicles if more are visible.\n\n"
                    f"IMPORTANT:\n"
                    f"- Be specific about make/model - use exact model names (e.g., '911 Turbo S', '430 Scuderia', '458 Italia')\n"
                    f"- For Ferrari models: Check badge text on detail crops. Badge '430'/'F430'/'Scuderia' = 430 Scuderia (NOT Enzo). "
                    f"Badge '458'/'Italia' = 458 Italia. Badge 'Enzo' = Enzo.\n"
                    f"- If badge text is unclear, use visual features: Enzo has gullwing doors and extreme wedge shape. "
                    f"430/458 have conventional doors.\n"
                    f"- Use the detail crops to check badges and identify all vehicles accurately.\n\n"
                    f"Provide ONLY the vehicle details in the format above, no thinking."
                )
            
            # Increase token limit for cloud providers and use only main image to avoid token issues
            if use_cloud:
                vehicles_response = chat(client, model_name, 
                                       [{'role': 'user', 'content': identify_prompt, 'images': [img_b64]}], 
                                       options_override={"num_predict": 400, "temperature": 0.1})
            else:
                vehicles_response = chat(client, model_name, 
                                       [{'role': 'user', 'content': identify_prompt, 'images': multi_vehicle_images}], 
                                       options_override={"num_predict": 200, "temperature": 0.1})
            
            vehicles_text = extract_message_text(vehicles_response) or ''
            _log(f"Vehicle identification content length: {len(vehicles_text)}, model_name: {model_name}, type: {type(model_name)}")
            # Fallback to thinking field for qwen3-vl
            if not vehicles_text and isinstance(model_name, str) and ('qwen3' in model_name.lower() or 'qwen-3vl' in model_name.lower()):
                vehicles_text = extract_message_thinking(vehicles_response) or ''
                _log(f"Vehicle identification using thinking field, length: {len(vehicles_text)}")
            elif not vehicles_text:
                _log(f"Vehicle identification content empty but model check failed: model_name={model_name}, is_str={isinstance(model_name, str)}")
            _log(f"Vehicle identification response: {vehicles_text[:300] if vehicles_text else 'EMPTY'}")
            
            # Parse vehicle details (simple key:value extraction)
            vehicle_blocks = vehicles_text.split('\n\n')
            for block in vehicle_blocks[:4]:  # Max 4 vehicles
                lines = [l.strip() for l in block.split('\n') if ':' in l]
                vehicle_data = {}
                for line in lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        key_clean = key.strip()
                        val_clean = val.strip()
                        # Normalize race number key (handle both 'RaceNumber' and 'Race Number')
                        if key_clean.lower() in ('racenumber', 'race number'):
                            vehicle_data['RaceNumber'] = val_clean  # Store as RaceNumber for consistency
                        else:
                            vehicle_data[key_clean] = val_clean
                if vehicle_data.get('Make') and vehicle_data.get('Model'):
                    vehicle_list.append(vehicle_data)
            
            _log(f"Identified {len(vehicle_list)} vehicles: {vehicle_list}")
            
            # Step 2: Generate rich motorsport scene description
            vehicle_summary = ", ".join([f"{v.get('Make', 'Unknown')} {v.get('Model', 'Unknown')}" for v in vehicle_list[:3]])
            if use_cloud:
                motorsport_prompt = (
                    f"Write a detailed description of this motorsport scene with {vc} vehicles"
                    f"{' featuring ' + vehicle_summary if vehicle_summary else ''}. "
                    f"Write in automotive journalist style (150-250 words). "
                    f"Focus on vehicles, positioning, race numbers/liveries, track environment, and atmosphere. "
                    f"Be descriptive and vivid. Provide ONLY the description."
                )
            else:
                motorsport_prompt = (
                    f"Write a detailed, engaging description of this motorsport scene featuring {vc} vehicles "
                    f"({vehicle_summary if vehicle_summary else 'multiple race cars'}). "
                    f"Write in an automotive journalist style (150-250 words). Focus on:\n"
                    "- The vehicles and their positioning\n"
                    "- Race numbers, liveries, and sponsors if visible\n"
                    "- Track environment and atmosphere\n"
                    "- Any notable details or action\n"
                    "Be descriptive and vivid. Provide ONLY the description, no thinking."
                )
            
            # Increase token limit for cloud providers
            scene_max_tokens = 600 if use_cloud else 350
            scene_response = chat(client, model_name, 
                               [{'role': 'user', 'content': motorsport_prompt, 'images': [img_b64]}], 
                               options_override={"num_predict": scene_max_tokens, "temperature": 0.3})
            
            scene_description = extract_message_text(scene_response) or ''
            _log(f"Scene description content length: {len(scene_description)}, model_name: {model_name}, type: {type(model_name)}")
            # Fallback to thinking field for qwen3-vl
            if not scene_description and isinstance(model_name, str) and ('qwen3' in model_name.lower() or 'qwen-3vl' in model_name.lower()):
                scene_description = extract_message_thinking(scene_response) or ''
                _log(f"Scene description using thinking field, length: {len(scene_description)}")
            elif not scene_description:
                _log(f"Scene description content empty but model check failed: model_name={model_name}, is_str={isinstance(model_name, str)}")
            _log(f"Scene description generated: {len(scene_description)} chars")
            
            # Step 3: Create comprehensive motorsport metadata
            parsed_motorsport = {
                'Image Type': 'Motorsport Scene',
                'Primary Subject': f'{vc} Racing Vehicles' + (f' ({vehicle_summary})' if vehicle_summary else ''),
                'Setting': 'Race Track / Motorsport Event',
                'Key Objects': ', '.join([f"{v.get('Make', '')} {v.get('Model', '')}" for v in vehicle_list]) if vehicle_list else 'Multiple race cars',
                'TagoMatic AI Summary': scene_description[:500],
                'LLM Backend': f"Ollama: {model_name}",
                'Vehicle Count': vc,
                'Scene Type': 'Multi-Vehicle Motorsport'
            }
            
            # Add primary vehicle (first identified) to Make/Model fields for compatibility
            if vehicle_list:
                parsed_motorsport['Make'] = vehicle_list[0].get('Make', 'Multi-Vehicle')
                parsed_motorsport['Model'] = vehicle_list[0].get('Model', 'Motorsport Scene')
                parsed_motorsport['Color'] = vehicle_list[0].get('Color', '')
                # Extract race number from first vehicle (handle both 'RaceNumber' and 'Race Number' keys)
                race_num = vehicle_list[0].get('RaceNumber') or vehicle_list[0].get('Race Number', '')
                if race_num and str(race_num).strip().lower() not in ('none', 'unknown', ''):
                    parsed_motorsport['Race Number'] = str(race_num).strip()
                    _log(f"[MULTI-VEHICLE] Extracted race number from primary vehicle: {parsed_motorsport['Race Number']}")
                else:
                    _log(f"[MULTI-VEHICLE] No race number found in primary vehicle (got: {race_num})")
                _log(f"Primary vehicle: {parsed_motorsport['Make']} {parsed_motorsport['Model']}")
            else:
                # Fallback if no vehicles identified
                parsed_motorsport['Make'] = 'Multi-Vehicle'
                parsed_motorsport['Model'] = 'Motorsport Scene'
                _log("No specific vehicles identified, using generic motorsport metadata")

            print(f"[DEBUG] Multi-vehicle scene description generated successfully")
            _log(f"Motorsport scene complete - returning metadata with {len(vehicle_list)} identified vehicles")
            return (scene_description or ''), parsed_motorsport
            
        except Exception as e:
            print(f"[DEBUG] Multi-vehicle scene generation failed: {e}")
            _log(f"Multi-vehicle scene generation failed: {e}")
            # Fall back to normal car analysis
            pass
    
    # OCR PASS: Extract text from image (badges, license plates, model numbers) before main analysis
    # This helps the LLM by providing explicit text context, improving accuracy for cases like 430 vs 458
    ocr_text = ""
    try:
        ocr_prompt = (
            "Extract ALL visible text from this image. Look CAREFULLY for:\n"
            "- Badge/model text on the car (e.g., 'Zonda', 'Huayra', 'Pagani', 'Sagaris', 'TVR', 'Koenigsegg', '430', 'F430', 'Scuderia', '458', '488', 'FF')\n"
            "- License plate numbers\n"
            "- Any brand names or model numbers visible\n"
            "- Check rear badges, side badges, doors, and any text on the car\n\n"
            "Return ONLY the text you see, one item per line. If no text visible, return 'No text found'.\n"
            "Be thorough - check all visible surfaces including small badges and emblems."
        )
        ocr_response = chat(client, model_name, 
                           [{'role': 'user', 'content': ocr_prompt, 'images': [img_b64]}],
                           options_override={"num_predict": 100, "temperature": 0.0})
        ocr_text = extract_message_text(ocr_response) or extract_message_thinking(ocr_response) or ""
        ocr_text = ocr_text.strip()
        if ocr_text and ocr_text.lower() != 'no text found':
            _log(f"OCR extracted text: {ocr_text[:200]}")
            # Add OCR context to prompt
            ocr_context = f"\n\nEXTRACTED TEXT FROM IMAGE:\n{ocr_text}\n\nIMPORTANT: Use this extracted text to identify the model. If you see:\n- 'Zonda' or 'Pagani' → Pagani Zonda\n- 'Huayra' → Pagani Huayra\n- 'Sagaris' or 'TVR' → TVR Sagaris\n- 'Koenigsegg' → Koenigsegg (specify model if visible)\n- '430'/'F430'/'Scuderia' → Ferrari 430 Scuderia\n- '458'/'Italia' → Ferrari 458 Italia\n- '488' → Ferrari 488 GTB/Spider\n- 'FF' → Ferrari FF\n\nBadge text is definitive for identification. Also verify body style matches the model."
        else:
            ocr_context = ""
            _log("OCR: No text found in image")
    except Exception as ocr_e:
        _log(f"OCR pass failed: {ocr_e}")
        ocr_context = ""
    
    # Proactively check user KB for hints (if no feedback context already)
    kb_hints = ""
    if not feedback_context:
        try:
            # Load both approved and user KBs (user takes precedence)
            approved_kb = load_approved_knowledge_base()
            user_kb = load_user_knowledge_base()
            
            # Merge KBs for hint collection (user overrides approved)
            # Pass empty dict as builtin since we only need approved+user for hints
            merged_kb = merge_knowledge_bases({}, approved_kb, user_kb) if approved_kb or user_kb else {}
            
            if merged_kb:
                # Collect all distinguishing cues from merged KB as hints
                # Group by make to detect multiple models
                hints_by_make = {}
                for make, models in merged_kb.items():
                    if isinstance(models, list):
                        for entry in models:
                            if isinstance(entry, dict):
                                car_model = entry.get('model', '')  # Use different variable name to avoid collision
                                cues = entry.get('distinguishing_cues', {})
                                if cues and any(desc for desc in cues.values() if desc):
                                    if make not in hints_by_make:
                                        hints_by_make[make] = []
                                    hints_by_make[make].append((car_model, cues))
                
                # Format hints - if multiple models for same make, emphasize differences
                hints_list = []
                for make, model_entries in hints_by_make.items():
                    if len(model_entries) > 1:
                        # Multiple models for same make - create comparison format
                        for car_model, cues in model_entries:
                            hint_parts = [f"{make} {car_model}:"]
                            # Emphasize unique features that distinguish this model from others
                            for angle, desc in cues.items():
                                if desc:
                                    hint_parts.append(f"{angle.capitalize()}: {desc}")
                            hints_list.append(" ".join(hint_parts))
                    else:
                        # Single model for this make - standard format
                        car_model, cues = model_entries[0]
                        hint_parts = [f"{make} {car_model}:"]
                        for angle, desc in cues.items():
                            if desc:
                                hint_parts.append(f"{angle.capitalize()}: {desc}")
                        hints_list.append(" ".join(hint_parts))
                
                if hints_list:
                    kb_hints = "\n\n" + "="*80 + "\n"
                    kb_hints += "⚠️ CRITICAL: USER VERIFIED KNOWLEDGE BASE ⚠️\n"
                    kb_hints += "="*80 + "\n"
                    kb_hints += "The following vehicles have been CORRECTLY IDENTIFIED by the user through manual verification.\n"
                    kb_hints += "These entries take ABSOLUTE PRIORITY over any built-in knowledge base.\n\n"
                    kb_hints += "IF THE CAR IN THE IMAGE MATCHES THESE DISTINGUISHING FEATURES, YOU MUST IDENTIFY IT AS THE SPECIFIED MAKE/MODEL.\n"
                    kb_hints += "DO NOT identify it as anything else, even if it looks similar to other models.\n\n"
                    
                    # Check if there are multiple models for any make
                    makes_with_multiple = {}
                    for h in hints_list:
                        if ':' in h:
                            parts = h.split(':', 1)
                            make_model = parts[0].strip()
                            make_model_parts = make_model.split()
                            if len(make_model_parts) >= 2:
                                make = make_model_parts[0]
                                if make not in makes_with_multiple:
                                    makes_with_multiple[make] = []
                                makes_with_multiple[make].append(h)
                    
                    # Add warning if multiple models for same make
                    for make, model_hints in makes_with_multiple.items():
                        if len(model_hints) > 1:
                            kb_hints += f"\n⚠️ CRITICAL: Multiple {make} models exist in USER KB. You MUST distinguish between them by comparing the UNIQUE features:\n"
                            # Extract model names for comparison
                            model_names = []
                            for hint in model_hints:
                                if ':' in hint:
                                    make_model = hint.split(':', 1)[0].strip()
                                    make_model_parts = make_model.split()
                                    if len(make_model_parts) >= 2:
                                        model_names.append(' '.join(make_model_parts[1:]))
                            if model_names:
                                kb_hints += f"Available {make} models in USER KB: {', '.join(model_names)}\n"
                                kb_hints += "Compare the EXHAUST, TAILIGHTS, REAR WING, and other unique features to determine which model it is.\n"
                                kb_hints += "The user has verified these identifications - trust their corrections over built-in data.\n"
                    
                    kb_hints += "\nUSER VERIFIED ENTRIES (check these FIRST - they override built-in KB):\n\n"
                    for hint in hints_list:
                        # Extract make/model from hint for emphasis
                        if ':' in hint:
                            make_model_part = hint.split(':', 1)[0].strip()
                            features_part = hint.split(':', 1)[1].strip() if ':' in hint else ""
                            kb_hints += f"✓ **{make_model_part}** - {features_part}\n"
                        else:
                            kb_hints += f"✓ {hint}\n"
                    # Add specific warning for each make in user KB
                    user_makes = set()
                    for hint in hints_list:
                        if ':' in hint:
                            make_model = hint.split(':', 1)[0].strip()
                            make = make_model.split()[0] if make_model.split() else ""
                            if make:
                                user_makes.add(make)
                    
                    for make in user_makes:
                        kb_hints += f"\n⚠️ IMPORTANT: User has verified a {make} model in their KB. "
                        kb_hints += f"If the {make} in the image matches the features described above, you MUST use the user-verified model name, "
                        kb_hints += f"NOT any built-in {make} model names.\n"
                    
                    kb_hints += "\n" + "="*80 + "\n"
                    
                    _log(f"[KB-HINTS] Injected {len(hints_list)} KB hints into prompt")
        except Exception as e:
            _log(f"Error collecting KB hints: {e}")
            import traceback
            _log(traceback.format_exc())
            kb_hints = ""
    
    # Use the new prompt template system
    context = {
        'use_cloud': use_cloud,
        'motorshow_mode': motorshow_mode,
        'race_mode': race_mode,
        'feedback': feedback_context,
        'weak_hint': weak_hint_text,
        'ocr_text': ocr_text if ocr_text else None,
        'kb_hints': kb_hints
    }
    
    _log(f"Context created: use_cloud={use_cloud}, motorshow={motorshow_mode}")
    print(f"[DEBUG] Context created: {context}")
    
    # Get appropriate prompt template
    print(f"[DEBUG] Calling get_prompt...")
    prompt = get_prompt('car_analysis', context)
    
    # KB hints are already added in get_prompt via context['kb_hints'], but we also append them here
    # for extra emphasis (they're added in get_prompt AND here to ensure they're visible)
    if kb_hints:
        prompt += kb_hints
    
    # Append OCR context if available
    if ocr_context:
        prompt += ocr_context
    
    _log(f"Prompt template retrieved, length: {len(prompt)}")
    print(f"[DEBUG] Prompt template retrieved, length: {len(prompt)}")
    
    # Track performance
    start_time = time.time()

    # Build images list; if cloud is active, keep crops smaller and in JPEG
    # Note: use_cloud is already set at the beginning of the function
    # For cloud, keep payload lean (main image only) to avoid output cap; add crops only for local
    # ONNX crops: dominant vehicle only, gets full + left + right (3 crops total)
    images_b64_list = [img_b64]
    if not use_cloud:
        try:
            _log(f"Generating detail crops for {image_path}")
            crops = generate_detail_crops(image_path, for_cloud=False)
            if crops:
                # Use only 3 crops for dominant vehicle (full + left + right)
                images_b64_list.extend(crops[:3])
                _log(f"Generated {len(crops[:3])} detail crops for dominant vehicle")
            else:
                _log(f"No detail crops generated, using main image only")
        except Exception as crop_error:
            _log(f"Detail crop generation failed: {crop_error}")
            import traceback
            _log(f"Crop generation traceback: {traceback.format_exc()}")
            # Continue with main image only if crops fail
            images_b64_list = [img_b64]
    
    _log(f"About to send car analysis prompt (model={model_name}, images={len(images_b64_list)})")
    try:
        # All models now use the same prompt from PROMPT_TEMPLATES
        # Token limits adjusted for cloud vs local
        if use_cloud:
            # Cloud providers need higher token limits for structured output
            response = chat(
                client,
                model_name,
                [{'role': 'user', 'content': prompt, 'images': images_b64_list}],
                options_override={
                    "temperature": (0.1 if motorshow_mode else 0.2),
                    "top_p": (0.7 if motorshow_mode else 0.8),
                    "num_predict": 600,  # Higher limit for cloud providers
                },
            )
        else:
            # Local models use smart retry with fallback
            response = smart_retry_with_fallback(client, model_name, prompt, img_b64, max_retries=2)
        
        # Record metrics
        response_time = time.time() - start_time
        success = bool(response and (str(response).strip() if isinstance(response, str) else True))
        prompt_metrics.record_prompt_result('car_analysis', success, response_time)
        
        if not success:
            prompt_metrics.record_fallback_usage('car_analysis')
            # Fallback to full images list
            # Increase token limit for cloud providers
            cloud_fallback_tokens = 500 if use_cloud else (160 if motorshow_mode else 120)
            response = chat(
                client,
                model_name,
                [{'role': 'user', 'content': prompt, 'images': images_b64_list}],
                options_override={
                    "temperature": (0.1 if motorshow_mode else 0.2),
                    "top_p": (0.7 if motorshow_mode else 0.8),
                    "num_predict": cloud_fallback_tokens,
                },
            )
    except Exception as e:
        prompt_metrics.record_fallback_usage('car_analysis')
        # Fallback to original error handling
        try:
            msg = str(e).lower()
            if any(tok in msg for tok in ['invalid jpeg', 'unexpected eof', '0xff00', 'failed to process inputs', 'status code: 500']):
                safe_b64 = encode_image_png_b64(image_path)
                cloud_error_tokens = 500 if use_cloud else (120 if motorshow_mode else 96)
                response = chat(
                    client,
                    model_name,
                    [{'role': 'user', 'content': prompt, 'images': [safe_b64]}],
                    options_override={
                        "temperature": (0.1 if motorshow_mode else 0.2),
                        "top_p": (0.6 if motorshow_mode else 0.8),
                        "num_predict": cloud_error_tokens,
                    },
                )
            else:
                cloud_error_tokens = 500 if use_cloud else (120 if motorshow_mode else 96)
                response = chat(
                    client,
                    model_name,
                    [{'role': 'user', 'content': prompt, 'images': [img_b64]}],
                    options_override={
                        "temperature": (0.1 if motorshow_mode else 0.2),
                        "top_p": (0.6 if motorshow_mode else 0.8),
                        "num_predict": cloud_error_tokens,
                    },
                )
        except Exception:
            # Final attempt with original bytes only
            cloud_final_tokens = 500 if use_cloud else (180 if motorshow_mode else 128)
            response = chat(
                client,
                model_name,
                [{'role': 'user', 'content': prompt, 'images': [img_b64]}],
                options_override={
                    "temperature": (0.1 if motorshow_mode else 0.2),
                    "top_p": (0.6 if motorshow_mode else 0.8),
                    "num_predict": cloud_final_tokens,
                },
            )

    _log(f"Car analysis response received from model {model_name}")
    primary_text = ''
    # Use extract_message_text for all responses (handles both Ollama and cloud formats)
    primary_text = extract_message_text(response)
    _log(f"Extracted content length: {len(primary_text) if primary_text else 0}")
    
    # If extraction failed or returned empty, try direct dict access as fallback
    if not primary_text or not str(primary_text).strip():
        try:
            if isinstance(response, dict) and 'message' in response:
                msg = response.get('message')
                if isinstance(msg, dict) and 'content' in msg:
                    primary_text = str(msg.get('content', '')).strip()
                    _log(f"Fallback extraction successful, length: {len(primary_text)}")
        except Exception as e:
            _log(f"Fallback extraction also failed: {e}")
    
    # Log actual response content if suspiciously short (helps debug cloud provider issues)
    if primary_text and len(primary_text) < 100:
        _log(f"WARNING: Response is suspiciously short ({len(primary_text)} chars). Content preview: {primary_text[:200]}")
    # If Qwen-3VL produced empty content but has thinking, extract from thinking
    # This is critical for qwen3 models which often put responses in thinking field
    try:
        is_qwen3_model = isinstance(model_name, str) and ('qwen3' in model_name.lower() or 'qwen-3vl' in model_name.lower())
        if (not primary_text or not str(primary_text).strip()) and is_qwen3_model:
            _log(f"Content is empty for qwen3 model ({model_name}), checking thinking field...")
            thinking = extract_message_thinking(response)
            _log(f"Thinking extracted, length: {len(thinking) if thinking else 0}")
            if thinking:
                _log(f"Thinking preview (first 300 chars): {thinking[:300]}")
            if isinstance(thinking, str) and thinking.strip():
                # First, try to extract directly from thinking using regex patterns
                _log(f"Attempting direct extraction from thinking field (qwen3-vl narrative parsing)")
                import re
                
                direct_extract = {}
                thinking_lower = thinking.lower()
                
                # Extract Make (look for "it's a [brand]" or "make: [brand]" or brand names)
                # Priority: explicit brand names first, then contextual patterns
                make_patterns = [
                    r'\b(Ferrari|Porsche|Bentley|Audi|BMW|Mercedes-Benz|Mercedes|Lamborghini|McLaren|Aston\s+Martin|Lotus|Jaguar|Alfa\s+Romeo|Maserati|Bugatti|Pagani|Koenigsegg|Tesla|Ford|Chevrolet|Dodge|Honda|Toyota|Nissan|Mazda|Subaru|Mitsubishi|Rolls-Royce|Lexus|Acura|Infiniti|Cadillac|Corvette)\b',
                    r"(?:make:\s*|it(?:'s|\s+is)\s+(?:a|an)\s+)([A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)",
                ]
                for pattern in make_patterns:
                    match = re.search(pattern, thinking, re.IGNORECASE)
                    if match:
                        make_candidate = match.group(1).strip() if hasattr(match, 'group') and match.lastindex else match.group(0).strip()
                        # Filter out false positives like "classic", "car", "vehicle", single letters
                        if (len(make_candidate) >= 3 and 
                            make_candidate.lower() not in {'classic', 'car', 'vehicle', 'sports', 'race', 'racing', 'let', 'the', 'and', 'for'}):
                            direct_extract['Make'] = make_candidate
                            break
                
                # Extract Model (look for specific model names or "model: [name]")
                # Priority: explicit model names first, then contextual patterns
                model_patterns = [
                    # Common model names
                    r'\b(Continental\s+GT3?|Aventador\s+SVJ?|Huracan\s+(?:STO|EVO)?|Artura|720S|765LT|P1|LaFerrari|458\s+(?:Italia|Speciale)?|488\s+(?:GTB|Pista)?|F8\s+Tributo|SF90|296\s+GTB|911\s+(?:GT3|Turbo|Carrera)?|Cayman|Boxster|Taycan|Panamera|R8|RS[34567]|TT|A[345678]|AMG\s+GT|SLS|SLR|C63|E63|S63|GT-R|NSX|Supra|RX-[78]|Corvette|Viper|Mustang|220\s+S|V12\s+Vantage|DB\d+|Vantage)\b',
                    # "model: [name]" or "model—[name]" patterns
                    r'model[:\s—]+(?:the\s+)?([A-Z][A-Za-z0-9\s]{2,25})(?:\.|,|\n|\()',
                    # "this is a [brand] [model]" pattern
                    r'(?:Mercedes-Benz|Mercedes|Aston\s+Martin)\s+([A-Z][A-Za-z0-9\s]{2,20})(?:\.|,)',
                ]
                for pattern in model_patterns:
                    match = re.search(pattern, thinking, re.IGNORECASE)
                    if match:
                        model_text = match.group(1).strip() if hasattr(match, 'group') and match.lastindex else match.group(0).strip()
                        # Clean up common prefixes/suffixes
                        model_text = re.sub(r'^(the\s+|a\s+)', '', model_text, flags=re.IGNORECASE)
                        model_text = re.sub(r'\s+(is|was|has|or|and)$', '', model_text, flags=re.IGNORECASE)
                        # Filter out junk words
                        if (len(model_text) >= 2 and 
                            model_text.lower() not in {'let', 'the', 'check', 'looking', 'analyze', 'first', 'model'}):
                            direct_extract['Model'] = model_text
                            break
                
                # Extract Color (look for "color is/dark gray/charcoal/red")
                color_patterns = [
                    r'color\s+is\s+([a-z\s]+?)(?:\.|,|\n|$)',
                    r'(?:dark|light|bright)?\s*(?:gray|grey|red|white|black|blue|yellow|green|silver|charcoal)',
                ]
                for pattern in color_patterns:
                    match = re.search(pattern, thinking, re.IGNORECASE)
                    if match:
                        if hasattr(match, 'group') and match.lastindex:
                            direct_extract['Color'] = match.group(1).strip()
                        else:
                            direct_extract['Color'] = match.group(0).strip()
                        break
                
                # Extract Logos (look for logo mentions and sponsor text)
                logos_found = []
                logo_patterns = [
                    r'["\']([A-Z]{2,}(?:\s+[A-Z]+)*)["\']',  # ALL CAPS text in quotes like "MJC", "GOODYEAR"
                    r'\b(MJC|GOODYEAR|MICHELIN|PIRELLI|SHELL|RED\s+BULL|MONSTER|MOBIL|CASTROL|PETRONAS|HANDLEY|TAYLOR|DELTA|INFINIT|MAGS\s+GROUP|ZERO\s+EMISSIONS|GT3|SVJ)\b',
                ]
                # Junk phrases to filter out
                junk_phrases = {'need to', 'let', 'on the', 'name', 'check all', 'visible', 'are on', 'list all', 'doors and sides', 'need to list', 'wait'}
                
                for pattern in logo_patterns:
                    matches = re.findall(pattern, thinking, re.IGNORECASE)
                    for match in matches:
                        logo = match.strip() if isinstance(match, str) else (match[0].strip() if isinstance(match, tuple) else str(match).strip())
                        logo_lower = logo.lower()
                        # Filter: must be 2-30 chars, not a junk phrase, not already added
                        if (2 <= len(logo) <= 30 and 
                            logo_lower not in junk_phrases and
                            not any(junk in logo_lower for junk in junk_phrases) and
                            logo not in logos_found):
                            logos_found.append(logo)
                
                if logos_found:
                    direct_extract['Logos'] = ', '.join(logos_found[:15])  # Up to 15 logos for motorsport
                
                # Extract Race Number (look for race number mentions or "43" pattern)
                race_num_patterns = [
                    r'race\s+number[:\s]+["\']?(\d+)["\']?',
                    r'["\'](\d{1,3})["\']',  # Numbers in quotes
                    r'(?:number|#)\s*(\d{1,3})',
                ]
                for pattern in race_num_patterns:
                    match = re.search(pattern, thinking, re.IGNORECASE)
                    if match:
                        direct_extract['Race Number'] = match.group(1).strip()
                        break
                
                # Extract License Plate (look for plate mentions)
                plate_patterns = [
                    r'(?:license|registration)\s+plate[:\s]+["\']?([A-Z0-9\s]+)["\']?',
                    r'(?:plate is|plate[:\s]+)["\']?([A-Z0-9\s]{3,12})["\']?',
                ]
                for pattern in plate_patterns:
                    match = re.search(pattern, thinking, re.IGNORECASE)
                    if match:
                        direct_extract['License Plate'] = match.group(1).strip()
                        break
                
                _log(f"Direct extraction results: {direct_extract}")
                
                # Try LLM reformat first (more accurate), fallback to direct extraction if it fails
                _log(f"Attempting LLM reformat first, with direct extraction as fallback")
                enforce_prompt = (
                    f"You previously analyzed this car and your analysis was:\n\n"
                    f"{thinking[:500]}\n\n"
                    f"Now provide ONLY the key values in this exact format (be brief):\n\n"
                    f"Make: [brand only]\n"
                    f"Model: [model only]\n"
                    f"Color: [color only]\n"
                    f"Logos: [list visible text/logos]\n"
                    f"Race Number: [number or Unknown]\n"
                    f"License Plate: [plate or Unknown]\n"
                    f"AI-Interpretation Summary: [detailed narrative description of the car, its features, setting, and any notable details - be descriptive and thorough, up to 500 chars]\n\n"
                    f"Answer with ONLY the 7 lines above. No extra text."
                )
                try:
                    restate = chat(
                        client,
                        model_name,
                        [
                            {'role': 'user', 'content': enforce_prompt, 'images': images_b64_list}
                        ],
                        options_override={"temperature": 0.0, "top_p": 0.7, "num_predict": 600},
                    )
                    primary_text = extract_message_text(restate)
                    _log(f"LLM reformat result, content length: {len(primary_text) if primary_text else 0}")
                    
                    # If content is still empty, check thinking field of reformat response (especially for qwen3)
                    if (not primary_text or not str(primary_text).strip()):
                        _log(f"Reformat content empty, checking reformat thinking field")
                        restate_thinking = extract_message_thinking(restate)
                        # For qwen3 models, if thinking has structured format, use it directly
                        if restate_thinking and is_qwen3_model:
                            _log(f"qwen3 reformat thinking found ({len(restate_thinking)} chars), checking for structured format")
                            # Look for structured lines in thinking
                            import re
                            structured_lines = []
                            for line in restate_thinking.split('\n'):
                                line = line.strip()
                                # Look for lines like "Make: Ferrari" or "Model: 430 Scuderia"
                                if ':' in line and any(k in line.lower()[:20] for k in ['make', 'model', 'color', 'logo', 'race', 'license', 'plate', 'summary']):
                                    # Skip template lines
                                    if '[' in line and ']' in line:
                                        continue
                                    structured_lines.append(line)
                            if structured_lines:
                                _log(f"Found {len(structured_lines)} structured lines in qwen3 thinking, using as primary_text")
                                primary_text = '\n'.join(structured_lines)
                        
                        # Fallback: if still no primary_text and we have thinking, try the existing parsing logic
                        if (not primary_text or not str(primary_text).strip()) and restate_thinking:
                            _log(f"Reformat thinking found, length: {len(restate_thinking)}")
                            # Try to extract structured lines from reformat thinking and clean them
                            import re
                            restate_lines = []
                            for line in restate_thinking.split('\n'):
                                line = line.strip()
                                if line and ':' in line and any(k in line.lower() for k in ['make', 'model', 'color', 'logo', 'race', 'license', 'plate', 'summary']):
                                    # Skip placeholder/template lines
                                    if '[' in line and ']' in line:
                                        _log(f"Skipping template line: {line[:80]}")
                                        continue
                                    # Clean the value: extract just the key info
                                    if ':' in line:
                                        field, value = line.split(':', 1)
                                        value = value.strip()
                                        
                                        # Remove quotes
                                        value = value.strip('"').strip("'")
                                        
                                        # Pattern 1: "so X" at end -> extract X
                                        so_match = re.search(r'so\s+["\']?([^"\'\.]+?)["\']?\.?$', value, re.IGNORECASE)
                                        if so_match:
                                            value = so_match.group(1).strip()
                                        else:
                                            # Pattern 2: "The X is Y, as seen..." -> extract Y
                                            is_as_match = re.search(r'\bis\s+([^,\.]+?)(?:,\s+as\s+)', value, re.IGNORECASE)
                                            if is_as_match:
                                                value = is_as_match.group(1).strip()
                                            else:
                                                # Pattern 3: "The X is Y." -> extract Y
                                                is_match = re.search(r'\bis\s+(?:a\s+|the\s+)?([A-Za-z0-9\s\-]+?)(?:\.|,|$)', value, re.IGNORECASE)
                                                if is_match:
                                                    value = is_match.group(1).strip()
                                                else:
                                                    # Pattern 4: Just take the last meaningful part after commas
                                                    parts = value.split(',')
                                                    if len(parts) > 1:
                                                        # Try last part first
                                                        last = parts[-1].strip()
                                                        if len(last) < 50 and not last.lower().startswith(('as ', 'so ', 'which ')):
                                                            value = last
                                        
                                        # Remove ", as seen/visible/shown..." at end
                                        value = re.sub(r',\s+(as|which|that)\s+(seen|visible|shown|found|present).*$', '', value, flags=re.IGNORECASE).strip()
                                        # Remove " color" at end (redundant)
                                        value = re.sub(r'\s+color$', '', value, flags=re.IGNORECASE).strip()
                                        # Remove common narrative prefixes
                                        value = re.sub(r'^(the\s+|a\s+)?(main\s+|primary\s+)?(color|make|model|brand)\s+(is\s+)?', '', value, flags=re.IGNORECASE).strip()
                                        # Remove parenthetical notes
                                        value = re.sub(r'\s*\([^)]*\)\s*', ' ', value).strip()
                                        # Remove trailing periods
                                        value = value.rstrip('.')
                                        # Capitalize first letter
                                        if value and len(value) > 0:
                                            value = value[0].upper() + value[1:] if len(value) > 1 else value.upper()
                                        # Reconstruct clean line
                                        line = f"{field.strip()}: {value}"
                                        _log(f"Cleaned line: {line[:100]}")
                                    restate_lines.append(line)
                            if restate_lines and len(restate_lines) >= 3:
                                primary_text = '\n'.join(restate_lines)
                                _log(f"Extracted {len(restate_lines)} lines from reformat thinking (cleaned)")
                    
                    if primary_text and len(primary_text.strip()) > 20:
                        _log(f"Successfully got concise response from LLM")
                        _log(f"Reformatted preview: {primary_text[:150]}")
                except Exception as e:
                    _log(f"LLM reformat failed: {e}")
                    pass
                # Final local normalization from thinking if content still empty
                if (not primary_text or not str(primary_text).strip()):
                    _log(f"Content still empty after restate, attempting smart extraction from thinking")
                    try:
                        # First try: structured labels (Make:, Model:, etc.)
                        low = thinking.lower()
                        if 'make:' in low and 'model:' in low:
                            start = low.find('make:')
                            # Map back to original case using the same index range
                            start_orig = thinking.lower().find('make:')
                            snippet = thinking[start_orig:]
                            lines = []
                            for ln in snippet.splitlines():
                                s = ln.strip()
                                if not s:
                                    continue
                                if any(s.lower().startswith(k) for k in ['make:', 'model:', 'color:', 'logos:', 'race number:', 'license plate:', 'ai-interpretation summary:']):
                                    lines.append(s)
                                if len(lines) >= 7:
                                    break
                            if not lines:
                                # Handle inline labels in a single sentence (e.g., "Make: X. Model: Y. Color: Z.")
                                import re
                                text = thinking
                                def grab(label):
                                    m = re.search(rf"{label}\s*:\s*(.*?)(?:[\n\r]|\.\s|$)", text, flags=re.IGNORECASE)
                                    return m.group(1).strip() if m else ''
                                
                                def clean_value(raw, field_type='generic'):
                                    """Extract concise value from verbose narrative text"""
                                    if not raw:
                                        return ''
                                    
                                    # First, remove non-ASCII/CJK characters (Chinese, Japanese, Korean, etc.)
                                    # Keep only Latin alphabet, numbers, punctuation, spaces
                                    raw = ''.join(c for c in raw if ord(c) < 128 or c in '—–')
                                    raw = raw.strip()
                                    if not raw:
                                        return ''
                                    
                                    # For summary, keep it as-is (narrative is fine)
                                    if field_type == 'summary':
                                        raw = raw.split('.')[0].strip()
                                        if len(raw) > 200:
                                            raw = raw[:197] + '...'
                                        return raw
                                    
                                    # For all other fields: extract KEY VALUE only
                                    # Pattern: "the car is BMW" or "so Make is BMW" -> extract "BMW"
                                    # Look for "is X" pattern
                                    is_match = re.search(r'\bis\s+(?:a\s+)?([A-Z][A-Za-z0-9\s\-]+?)(?:\.|,|$|\s+\()', raw)
                                    if is_match:
                                        return is_match.group(1).strip()
                                    
                                    # Pattern: "has X" -> extract "X"
                                    has_match = re.search(r'\bhas\s+(?:the\s+)?([A-Z][A-Za-z0-9\s\-]+?)(?:\.|,|$|\s+\()', raw)
                                    if has_match:
                                        return has_match.group(1).strip()
                                    
                                    # Pattern: "include X" -> extract "X"
                                    include_match = re.search(r'\binclude[s]?\s+(?:the\s+)?([A-Z][A-Za-z0-9\s\-,]+?)(?:\.|$|\s+\()', raw)
                                    if include_match:
                                        return include_match.group(1).strip()
                                    
                                    # Pattern: quoted value "X" -> extract X
                                    quote_match = re.search(r'["\']([^"\']+)["\']', raw)
                                    if quote_match:
                                        return quote_match.group(1).strip()
                                    
                                    # Fallback: take last capitalized word/phrase
                                    words = raw.split()
                                    for i in range(len(words)-1, -1, -1):
                                        if words[i] and words[i][0].isupper():
                                            # Found a capitalized word, take from here to end (max 4 words)
                                            result = ' '.join(words[i:i+4])
                                            # Clean punctuation
                                            result = re.sub(r'[,;:.]$', '', result)
                                            return result[:80]
                                    
                                    # Last resort: just clean and truncate
                                    raw = re.sub(r'^(the|it\'s|so|but)\s+', '', raw, flags=re.IGNORECASE)
                                    return raw[:80].strip()
                                
                                mk = clean_value(grab('Make'), 'make')
                                md = clean_value(grab('Model'), 'model')
                                co = clean_value(grab('Color'), 'color')
                                lg = clean_value(grab('Logos?') or grab('Logo'), 'logos')
                                rn = clean_value(grab(r'Race\s*Number'), 'race_num')
                                lp = clean_value(grab(r'License\s*Plate'), 'plate')
                                sm = clean_value(grab(r'AI-Interpretation\s*Summary'), 'summary')
                                cand = []
                                cand.append(f"Make: {mk or 'Unknown'}")
                                cand.append(f"Model: {md or 'Unknown'}")
                                cand.append(f"Color: {co or 'Unknown'}")
                                cand.append(f"Logos: {lg or ''}")
                                cand.append(f"Race Number: {rn or 'Unknown'}")
                                cand.append(f"License Plate: {lp or 'Unknown'}")
                                cand.append(f"AI-Interpretation Summary: {sm or ''}")
                                lines = cand
                            if lines:
                                primary_text = '\n'.join(lines)
                                _log(f"Successfully extracted {len(lines)} lines from thinking inline labels")
                                _log(f"Extracted lines preview: {primary_text[:200]}")
                        
                        # Second try: narrative extraction from thinking
                        if (not primary_text or not str(primary_text).strip()):
                            _log(f"Trying narrative extraction from thinking")
                            import re
                            # Look for car make/model in narrative text
                            # Pattern: "the car is a BMW M4" or "BMW M4" or "it's a BMW... M4"
                            make_keywords = ['bmw', 'mercedes', 'audi', 'porsche', 'ferrari', 'lamborghini', 'mclaren', 'aston martin', 'jaguar', 'ford', 'chevrolet', 'dodge', 'toyota', 'nissan', 'honda', 'mazda', 'subaru', 'volkswagen', 'volvo', 'alfa romeo', 'lotus', 'morgan', 'caterham', 'radical']
                            thinking_lower = thinking.lower()
                            found_make = None
                            found_model = None
                            found_color = None
                            found_plate = None
                            
                            # Find make
                            for mk in make_keywords:
                                if mk in thinking_lower:
                                    # Get the original case version
                                    idx = thinking_lower.find(mk)
                                    found_make = thinking[idx:idx+len(mk)].title()
                                    break
                            
                            # Find model after make (look for alphanumeric patterns)
                            if found_make:
                                make_idx = thinking_lower.find(found_make.lower())
                                after_make = thinking[make_idx + len(found_make):make_idx + len(found_make) + 100]
                                # Look for model patterns: M4, M3, 911, GT3, F40, etc.
                                model_match = re.search(r'\b([A-Z0-9]{1,4}(?:\s?[A-Z]{1,3})?)\b', after_make)
                                if model_match:
                                    found_model = model_match.group(1).strip()
                            
                            # Find color
                            color_keywords = ['yellow', 'gold', 'red', 'blue', 'green', 'black', 'white', 'silver', 'gray', 'grey', 'orange', 'purple', 'brown']
                            for col in color_keywords:
                                if col in thinking_lower:
                                    found_color = col.capitalize()
                                    break
                            
                            # Find license plate (pattern: 2-4 chars, optional space, 2-4 chars/digits)
                            plate_match = re.search(r'\b([A-Z]{1,4}[0-9]{1,4}\s?[A-Z]{1,4})\b', thinking, re.IGNORECASE)
                            if plate_match:
                                found_plate = plate_match.group(1).upper()
                            
                            if found_make or found_model:
                                _log(f"Narrative extraction: Make={found_make}, Model={found_model}, Color={found_color}, Plate={found_plate}")
                                lines = []
                                lines.append(f"Make: {found_make or 'Unknown'}")
                                lines.append(f"Model: {found_model or 'Unknown'}")
                                lines.append(f"Color: {found_color or 'Unknown'}")
                                lines.append(f"Logos: ")
                                lines.append(f"Race Number: Unknown")
                                lines.append(f"License Plate: {found_plate or 'Unknown'}")
                                summary = f"{found_make or 'Car'} {found_model or ''} in {found_color or 'unknown'} color".strip()
                                lines.append(f"AI-Interpretation Summary: {summary}")
                                primary_text = '\n'.join(lines)
                                _log(f"Successfully extracted from narrative thinking")
                    except Exception as e:
                        _log(f"Smart extraction failed: {e}")
                        pass
    except Exception as e:
        _log(f"Thinking extractor block failed: {e}")
        pass
    # If the provider returned only meta JSON (no usable content), treat as empty to trigger retries/parsers
    try:
        if _looks_meta_only_json(primary_text):
            primary_text = ''
    except Exception:
        pass
    # If the analysis returned a counting-style payload (Cars:/Motorcycles:/Total:), re-ask once with strict 7-line prompt
    try:
        pt_low = (primary_text or '').lower()
        if ('cars:' in pt_low) and ('motorcycles:' in pt_low) and ('total:' in pt_low):
            _log("Car analysis returned counting payload; re-asking with strict 7-line format")
            strict_prompt = (
                "Identify this car strictly from the image. Return ONLY these 7 lines and nothing else.\n"
                "Make: <brand or Unknown>\n"
                "Model: <model or Unknown>\n"
                "Color: <color or Unknown>\n"
                "Logos: <logos/text or Unknown>\n"
                "Race Number: <number or Unknown>\n"
                "License Plate: <plate or Unknown>\n"
                "AI-Interpretation Summary: <descriptive narrative, up to 500 characters>"
            )
            try:
                # Increase token limit for cloud providers when retrying
                retry_tokens = 600 if use_cloud else 220
                resp_fix = chat(
                    client,
                    model_name,
                    [{'role': 'user', 'content': strict_prompt, 'images': [img_b64]}],
                    options_override={"temperature": 0.0, "num_predict": retry_tokens}
                )
                fixed_text = extract_message_text(resp_fix) or ''
                if fixed_text and ('make:' in fixed_text.lower()) and ('model:' in fixed_text.lower()):
                    primary_text = fixed_text
            except Exception as _e_fix:
                _log(f"Strict re-ask failed: {_e_fix}")
    except Exception:
        pass

    parsed = parse_results_lines(primary_text)
    # Merge with a more permissive fallback parser to avoid losing good fields
    try:
        alt = parse_or_fallback_json(primary_text)
        if isinstance(alt, dict):
            for key in ('Make','Model','Color','Logos','Race Number','License Plate','AI-Interpretation Summary'):
                cur = str((parsed or {}).get(key, '')).strip()
                altv = str(alt.get(key, '')).strip()
                if not cur and altv:
                    parsed[key] = altv
    except Exception:
        pass
    
    # FALLBACK: If parsing failed and we have direct_extract data, use it
    try:
        if 'direct_extract' in locals() and direct_extract:
            # Check if parsed is empty or has mostly unknown values
            parsed_empty = not parsed or all(
                str(parsed.get(k, '')).strip().lower() in {'', 'unknown', 'n/a', 'none'} 
                for k in ('Make', 'Model', 'Color')
            )
            if parsed_empty:
                _log(f"Parsing failed, using direct extraction as fallback")
                # Build structured output from direct extraction
                parsed = parsed or {}
                if 'Make' in direct_extract and direct_extract['Make']:
                    parsed['Make'] = direct_extract['Make']
                if 'Model' in direct_extract and direct_extract['Model']:
                    parsed['Model'] = direct_extract['Model']
                if 'Color' in direct_extract and direct_extract['Color']:
                    parsed['Color'] = direct_extract['Color']
                if 'Logos' in direct_extract and direct_extract['Logos']:
                    parsed['Logos'] = direct_extract['Logos']
                if 'Race Number' in direct_extract and direct_extract['Race Number']:
                    parsed['Race Number'] = direct_extract['Race Number']
                if 'License Plate' in direct_extract and direct_extract['License Plate']:
                    parsed['License Plate'] = direct_extract['License Plate']
                # Use first 300 chars of thinking as summary
                if 'thinking' in locals() and thinking:
                    summary = thinking[:300].replace('\n', ' ').strip()
                    parsed['AI-Interpretation Summary'] = summary
                _log(f"Direct extraction fallback applied: Make={parsed.get('Make')}, Model={parsed.get('Model')}")
    except Exception as fallback_error:
        _log(f"Direct extraction fallback failed: {fallback_error}")
        pass
    
    # MODEL VERIFICATION: Check if model name contains suspicious OCR errors
    # (e.g., "2000 RC" instead of "ZEOD RC", "ZOod" instead of "ZEOD")
    try:
        model_val = str(parsed.get('Model', '')).strip() if parsed else ''
        make_val = str(parsed.get('Make', '')).strip() if parsed else ''
        
        # Detect suspicious patterns in model name
        suspicious_model = False
        if model_val and model_val.lower() not in {'unknown', 'n/a', 'none'}:
            # Pattern 1: Contains "00" or "000" (often OCR errors for "OO" or letters)
            if '00' in model_val:
                suspicious_model = True
                _log(f"Suspicious model detected (contains '00'): {model_val}")
            # Pattern 2: Looks like OCR confusion (e.g., "ZOod", "2Ood", "200d")
            import re
            if re.search(r'[0O][0O]d|[Z2][0O]od|RC\s*[HY]{2}', model_val, re.IGNORECASE):
                suspicious_model = True
                _log(f"Suspicious model detected (OCR pattern): {model_val}")
            # Pattern 3: Common Ferrari model confusions (Enzo vs LaFerrari vs 430 Scuderia)
            # These look similar and LLM often confuses them - verify by design
            # NOTE: Disabled verification for Ferrari 430/458 since we now have specific badge/exhaust prompt
            # The verification was actually making things worse by overriding correct answers
            if make_val and make_val.lower() == 'ferrari':
                if re.search(r'\b(enzo|laferrari|488)\b', model_val, re.IGNORECASE):
                    # Only verify Enzo/LaFerrari/488, not 430/458 (we have specific prompt for those)
                    suspicious_model = True
                    _log(f"Suspicious Ferrari model detected (common confusion): {model_val}")
                elif re.search(r'\b(430|scuderia|458)\b', model_val, re.IGNORECASE):
                    # Skip verification for 430/458 - our enhanced prompt with badge/exhaust rules handles this
                    suspicious_model = False
                    _log(f"Ferrari 430/458 detected - using enhanced prompt, skipping verification")
        
        # If suspicious, ask LLM to verify by analyzing VISUAL DESIGN, not just text
        if suspicious_model and make_val:
            _log(f"Verifying model name - analyzing visual design features")
            verify_prompt = (
                f"Look at this {make_val} car and analyze its VISUAL DESIGN FEATURES to identify the model.\n\n"
                f"DO NOT rely on text, logos, or previous readings. Analyze these design elements:\n"
                f"- Front grille: shape, size, position\n"
                f"- Headlights: design, position, shape\n"
                f"- Body proportions: overall silhouette\n"
                f"- Hood design: contours, vents, shape\n"
                f"- Side air intakes: size, shape, position\n"
                f"- Rear wing: design, position (if visible)\n\n"
                f"SPECIAL ATTENTION for Ferrari models:\n"
                f"- Enzo: More angular, wedge-shaped front, distinctive side air intakes, older design\n"
                f"- LaFerrari: More modern, smoother lines, hybrid-era design, different headlight shape\n"
                f"- 430 Scuderia: Mid-engined, different proportions than Enzo/LaFerrari\n"
                f"- 458 Italia: Different front grille, more rounded than Enzo\n\n"
                f"Previous reading was '{model_val}' - verify if this matches the actual design.\n\n"
                f"Based on VISUAL FEATURES only, what is the model?\n"
                f"Provide ONLY the model name (e.g., 'ZEOD RC', '430 Scuderia', 'Enzo', 'LaFerrari', '458 Italia')."
            )
            try:
                verify_resp = chat(
                    client,
                    model_name,
                    [{'role': 'user', 'content': verify_prompt, 'images': images_b64_list}],
                    options_override={"num_predict": 150, "temperature": 0.0}
                )
                verify_text = extract_message_text(verify_resp) or extract_message_thinking(verify_resp) or ''
                verify_text = verify_text.strip()
                
                _log(f"Model verification response: {verify_text[:200]}")
                
                # Extract corrected model name from response (handle both short and verbose responses)
                corrected_model = None
                
                # Pattern 1: Look for explicit model names (430 Scuderia, Enzo, ZEOD RC, etc.)
                # Prioritize actual known model names
                known_models = [
                    r'\b(430\s+Scuderia|458\s+Italia|488\s+GTB|F8\s+Tributo|Enzo|LaFerrari|SF90|296\s+GTB)\b',
                    r'\b(ZEOD\s+RC|DeltaWing|GT-R|GT3|Artura|720S|Aventador|Huracan)\b',
                    r'\b(911|Cayman|Boxster|Taycan|Panamera|Continental\s+GT3)\b',
                ]
                for pattern in known_models:
                    model_match = re.search(pattern, verify_text, re.IGNORECASE)
                    if model_match:
                        corrected_model = model_match.group(1).strip()
                        _log(f"Found known model pattern: {corrected_model}")
                        break
                
                # Pattern 2: Look for explicit corrections like "The actual text is ZEOD" or "It should be ZEOD RC"
                if not corrected_model:
                    corrected_match = re.search(r'(?:actual|correct|should be|text is|says|model is|this is)\s+["\']?([A-Z0-9]{2,}(?:\s+[A-Z0-9]{2,}){0,2})["\']?', verify_text, re.IGNORECASE)
                    if corrected_match:
                        corrected_model = corrected_match.group(1).strip()
                        _log(f"Found correction pattern: {corrected_model}")
                
                # Pattern 3: Look for quoted text like "ZEOD RC" or 'ZEOD'
                if not corrected_model:
                    quoted_match = re.search(r'["\']([A-Z]{2,}(?:[A-Z0-9\s\-]{0,20})?)["\']', verify_text)
                    if quoted_match:
                        corrected_model = quoted_match.group(1).strip()
                        _log(f"Found quoted text: {corrected_model}")
                
                # Pattern 4: Look for "ZEOD" or other ALL CAPS words (likely the correct model name)
                if not corrected_model:
                    caps_matches = re.findall(r'\b([A-Z]{3,}(?:\s+[A-Z]{2,})?)\b', verify_text)
                    for caps_word in caps_matches:
                        # Skip common words and the original wrong reading
                        if caps_word not in {'THE', 'AND', 'BUT', 'FOR', 'NOT', 'WAIT', 'LET', 'SEE', 'TEXT', 'READ', model_val.upper()}:
                            # Prioritize known correct model names (ZEOD, Scuderia, etc.)
                            if caps_word in {'ZEOD', 'SCUDERIA', 'AVENTADOR', 'ENZO', 'GT3', 'GT-R'}:
                                corrected_model = caps_word.strip()
                                _log(f"Found known model name: {corrected_model}")
                                break
                            elif not corrected_model:  # Take first valid match if no known model found
                                corrected_model = caps_word.strip()
                                _log(f"Found ALL CAPS word: {corrected_model}")
                    
                    # Also check for "ZEOD RC" pattern specifically
                    zeod_match = re.search(r'\bZEOD\s+RC\b', verify_text, re.IGNORECASE)
                    if zeod_match:
                        corrected_model = 'ZEOD RC'
                        _log(f"Found ZEOD RC pattern: {corrected_model}")
                
                # Pattern 5: If response is short (< 50 chars) and different from original, use it
                if not corrected_model and len(verify_text) < 50:
                    # Clean up the response
                    cleaned = re.sub(r'^(the\s+|it\s+is\s+|should\s+be\s+|actually\s+)', '', verify_text, flags=re.IGNORECASE).strip()
                    if len(cleaned.split()) <= 4 and cleaned.lower() not in {'correct', 'yes', 'confirmed', 'that is correct'}:
                        corrected_model = cleaned.strip().strip('"').strip("'")
                        _log(f"Using short response: {corrected_model}")
                
                # Apply correction if found and different from original
                if corrected_model and corrected_model.lower() != model_val.lower():
                    _log(f"Model corrected: '{model_val}' → '{corrected_model}'")
                    parsed['Model'] = corrected_model
                else:
                    _log(f"No correction needed or correction same as original")
            except Exception as verify_error:
                _log(f"Model verification failed: {verify_error}")
                pass
            
            # PATH-BASED HINT: If verification failed but path contains model name, use it
            if not corrected_model or corrected_model == model_val:
                try:
                    import re as re_path
                    path_lower = image_path.lower()
                    # Look for model names in path (e.g., "nissan-zeod-rc" or "zeod-rc")
                    path_models = {
                        'zeod': 'ZEOD RC',
                        'zeod-rc': 'ZEOD RC',
                        'zeod_rc': 'ZEOD RC',
                        'scuderia': 'Scuderia',
                        'aventador': 'Aventador',
                        'enzo': 'Enzo',
                        'laferrari': 'LaFerrari',
                        'la-ferrari': 'LaFerrari',
                        'la_ferrari': 'LaFerrari',
                    }
                    for path_key, model_name in path_models.items():
                        if path_key in path_lower and make_val.lower() in path_lower:
                            # Path contains model hint - use it if:
                            # 1. Current model is suspicious (OCR errors)
                            # 2. OR current model is different from path hint (common confusion like Enzo vs LaFerrari)
                            model_val_lower = model_val.lower()
                            path_model_lower = model_name.lower()
                            
                            if ('00' in model_val or 
                                re_path.search(r'[0O][0O]d|[Z2][0O]od', model_val, re_path.IGNORECASE) or
                                (model_val_lower != path_model_lower and 
                                 any(confused in model_val_lower for confused in ['enzo', 'laferrari', '430', 'scuderia', '458']))):
                                _log(f"Path hint correction: '{model_val}' → '{model_name}' (from path: {path_key})")
                                parsed['Model'] = model_name
                                corrected_model = model_name
                                break
                except Exception as path_hint_error:
                    _log(f"Path hint correction failed: {path_hint_error}")
                    pass
    except Exception as model_verify_error:
        _log(f"Model verification block failed: {model_verify_error}")
        pass
    
    # CRITICAL FIX: Don't aggressively extract car fields for non-car images
    # Only try car extraction if we actually have car fields
    try:
        has_car_fields = any(str(v or '').strip() for k, v in (parsed or {}).items() if k in ['Make','Model','Color','Logos','Race Number','License Plate'])
        if has_car_fields:
            # Only do aggressive car extraction if we already have some car data
            pass
    except Exception:
        pass
    
    # Return the analysis results - CRITICAL: ensure we always return a tuple
    _log(f"infer_enhanced returning: raw_text_length={len(primary_text) if primary_text else 0}, parsed_keys={list((parsed or {}).keys())}")
    return (primary_text or ''), (parsed or {})

def write_metadata_to_image(image_path: str, metadata: dict) -> bool:
    """Write EXIF UserComment (JSON) and IPTC fields using exiftool.
    
    This function writes FRESH metadata generated from current analysis.
    It completely ignores any existing metadata in the image.
    
    Supports all image formats that exiftool can write to:
    JPEG, TIFF, PNG, GIF, RAW formats (CR2, NEF, ARW, etc.), HEIC, WebP, etc.
    
    Hardened for PyInstaller EXE: uses _find_exiftool(), hides console window,
    longer timeouts, and logs detailed diagnostics to log.log so failures are visible.
    """
    try:
        # Let exiftool handle format detection - it supports many formats
        # (JPEG, TIFF, PNG, GIF, RAW, HEIC, WebP, etc.)

        def _log(msg: str) -> None:
            try:
                with open(os.path.join(os.getcwd(), 'log.log'), 'a', encoding='utf-8') as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [EXIFWRITE] {msg}\n")
            except Exception:
                pass

        # Sanitize values
        def _clean_str(v: str) -> str:
            try:
                s = str(v or '').strip()
                if s.startswith('**'):
                    s = s.lstrip('* ').strip()
                if s.endswith('**'):
                    s = s.rstrip('* ').strip()
                s = s.strip('"')
                # Remove non-ASCII/CJK characters (keep only Latin, numbers, punctuation, spaces)
                # This filters out Chinese, Japanese, Korean, Arabic, Cyrillic, etc.
                s = ''.join(c for c in s if ord(c) < 128 or c in '—–')  # Keep ASCII + common dashes
                return s.strip()
            except Exception:
                return str(v)

        meta = dict(metadata or {})
        for k in list(meta.keys()):
            kl = str(k).lower()
            if "here's a detailed analysis" in kl:
                meta.pop(k, None)
        
        # Check if this is a non-car image (has NC_* fields OR new non-car field names)
        is_non_car = any(k.startswith('NC_') for k in meta.keys()) or any(k in ['Image Type', 'Primary Subject', 'Setting', 'Key Objects', 'Descriptive Summary'] for k in meta.keys())
        
        # If non-car image, clear all car-specific fields
        if is_non_car:
            car_fields_to_clear = ['Make', 'Model', 'Color', 'Logos', 'License Plate']
            for field in car_fields_to_clear:
                meta.pop(field, None)

            # Normalize and ensure NC_* keys exist for consistent EXIF JSON
            try:
                if 'NC_ImageType' not in meta:
                    meta['NC_ImageType'] = _clean_str(meta.get('Image Type', 'Photograph'))
                if 'NC_Primary' not in meta:
                    meta['NC_Primary'] = _clean_str(meta.get('Primary Subject', ''))
                if 'NC_Setting' not in meta:
                    meta['NC_Setting'] = _clean_str(meta.get('Setting', ''))
                if 'NC_KeyObjects' not in meta:
                    meta['NC_KeyObjects'] = _clean_str(meta.get('Key Objects', ''))
            except Exception:
                pass
        
        for key in ['Make', 'Model', 'Color', 'Logos', 'License Plate', 'AI-Interpretation Summary', SUMMARY_KEY, 'LLM Backend', 'Second Pass']:
            if key in meta and isinstance(meta[key], str):
                meta[key] = _clean_str(meta[key])
        for noise_key in list(meta.keys()):
            nk = str(noise_key).lower()
            if nk in {"finishreason", "responseid", "modelversion"} or nk.startswith('{"finishreason"'):
                meta.pop(noise_key, None)
        # Normalize legacy summary key -> unified SUMMARY_KEY
        try:
            legacy = meta.get('AI-Interpretation Summary')
            if SUMMARY_KEY not in meta and legacy:
                meta[SUMMARY_KEY] = legacy
                meta.pop('AI-Interpretation Summary', None)
        except Exception:
            pass

        # Locate exiftool within PyInstaller layout
        exe = _find_exiftool()
        if not exe:
            _log('exiftool not found')
            return False

        import subprocess, tempfile
        kwargs = _exiftool_run_kwargs(exe, timeout=30)

        # Write EXIF UserComment via temp file to avoid quoting issues
        metadata_str = json.dumps(meta, ensure_ascii=False)
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as tf:
            tf.write(metadata_str)
            tf_path = tf.name
        cmd1 = [exe, '-m', '-charset', 'UTF8', '-charset', 'iptc=UTF8', '-charset', 'filename=UTF8', '-overwrite_original', f'-EXIF:UserComment<={tf_path}', str(image_path)]
        r1 = subprocess.run(cmd1, **kwargs)
        _log(f'rc1={r1.returncode} exe={exe} img={image_path}')
        if r1.stdout:
            _log(f'stdout1={r1.stdout.strip()}')
        if r1.stderr:
            _log(f'stderr1={r1.stderr.strip()}')
        # Handle stale exiftool temp file blocking writes on network paths
        if r1.returncode != 0 and r1.stderr and 'Temporary file already exists' in r1.stderr:
            try:
                p = str(image_path)
                candidates = [f"{p}_exiftool_tmp"]
                # Try both slash variants just in case
                candidates.append(f"{p.replace('\\', '/')}_exiftool_tmp")
                candidates.append(f"{p.replace('/', '\\')}_exiftool_tmp")
                for tpath in candidates:
                    try:
                        if os.path.exists(tpath):
                            os.remove(tpath)
                    except Exception:
                        pass
                # Retry once after cleanup
                r1 = subprocess.run(cmd1, **kwargs)
                _log(f'retry_rc1={r1.returncode}')
                if r1.stdout:
                    _log(f'retry_stdout1={r1.stdout.strip()}')
                if r1.stderr:
                    _log(f'retry_stderr1={r1.stderr.strip()}')
            except Exception:
                pass
        ok = (r1.returncode == 0)

        # IPTC helpful fields
        if ok:
            kws: list[str] = []
            make = _clean_str(meta.get('Make', ''))
            model = _clean_str(meta.get('Model', ''))
            color = _clean_str(meta.get('Color', ''))
            plate = _clean_str(meta.get('License Plate', ''))
            logos = _clean_str(meta.get('Logos', ''))
            race_num = _clean_str(meta.get('Race Number', ''))
            summary = _clean_str(meta.get('TagoMatic AI Summary', meta.get('AI-Interpretation Summary', '')))
            
            # Check if this is a non-car image (has NC_* fields OR new non-car field names)
            is_non_car = any(k.startswith('NC_') for k in meta.keys()) or any(k in ['Image Type', 'Primary Subject', 'Setting', 'Key Objects', 'Descriptive Summary'] for k in meta.keys())
            
            # LLM-BASED KEYWORD GENERATION: Ask LLM to generate comprehensive keyword list
            # This is more intelligent than simple string splitting
            llm_keywords = []
            if not is_non_car and (make or model or logos):
                try:
                    # Get client if available for keyword generation
                    llm_backend = meta.get('LLM Backend', '')
                    if 'ollama' in llm_backend.lower() or 'qwen' in llm_backend.lower():
                        try:
                            import ollama
                            kw_client = ollama.Client(host='http://127.0.0.1:11434', timeout=30)
                            
                            # Build keyword generation prompt
                            kw_prompt = (
                                f"Generate a comprehensive list of keywords for image search/catalog based on this car:\n\n"
                                f"Make: {make or 'Unknown'}\n"
                                f"Model: {model or 'Unknown'}\n"
                                f"Color: {color or 'Unknown'}\n"
                                f"Logos/Sponsors: {logos or 'None'}\n"
                                f"Race Number: {race_num or 'None'}\n"
                                f"Summary: {summary[:150] if summary else 'None'}\n\n"
                                f"Provide 10-20 relevant keywords as a comma-separated list. Include:\n"
                                f"- Car brand and model\n"
                                f"- Color\n"
                                f"- ALL sponsor/logo brands mentioned\n"
                                f"- Race number if present\n"
                                f"- Car type/category (e.g., sports car, race car, GT3, supercar)\n"
                                f"- Relevant motorsport terms if applicable\n\n"
                                f"Provide ONLY the comma-separated keyword list, no explanations."
                            )
                            
                            # Use qwen3-vl or fallback model for keyword generation
                            kw_model = llm_backend.split(':')[-1].strip() if ':' in llm_backend else 'qwen3-vl:latest'
                            kw_response = kw_client.chat(
                                model=kw_model,
                                messages=[{'role': 'user', 'content': kw_prompt}],
                                options={"num_predict": 150, "temperature": 0.2}
                            )
                            
                            # Extract keywords from response
                            kw_text = ''
                            if hasattr(kw_response, 'message') and hasattr(kw_response.message, 'content'):
                                kw_text = kw_response.message.content or ''
                            if not kw_text and hasattr(kw_response, 'message') and hasattr(kw_response.message, 'thinking'):
                                kw_text = kw_response.message.thinking or ''
                            
                            if kw_text:
                                # Parse comma-separated keywords
                                import re
                                kw_parts = re.split(r'[,;]+', kw_text)
                                for kw in kw_parts:
                                    kw = kw.strip().strip('"').strip("'").strip()
                                    if 2 <= len(kw) <= 30 and kw.lower() not in {'keywords', 'here', 'are', 'the', 'list'}:
                                        llm_keywords.append(kw)
                                
                                _log(f"LLM generated {len(llm_keywords)} keywords: {', '.join(llm_keywords[:10])}")
                        except Exception as kw_error:
                            _log(f"LLM keyword generation failed: {kw_error}")
                            pass
                except Exception:
                    pass
            
            if not is_non_car:
                # Priority 1: LLM-generated keywords (most comprehensive)
                if llm_keywords:
                    kws.extend(llm_keywords[:15])  # Top 15 LLM keywords
                
                # Priority 2: Core car identifiers (ensure they're included)
                if make and make not in kws:
                    kws.append(make)
                if model and model not in kws:
                    kws.append(model)
                if color and color not in kws:
                    kws.append(color)
                if plate and plate not in kws:
                    kws.append(plate)
                    
                # Priority 3: Add logo brands if available - be thorough!
                if logos:
                    # Extract brand names from logos string (support comma/semicolon/slash separators)
                    import re
                    # Split on commas, semicolons, slashes, pipes
                    logo_parts = re.split(r'[,;/|]+', logos)
                    for part in logo_parts:
                        part = part.strip().strip('"').strip("'").strip()
                        # Keep meaningful brand/sponsor names (2-30 chars, not just numbers)
                        if 2 <= len(part) <= 30 and not part.isdigit() and part.lower() not in {'n/a', 'none', 'unknown', ''}:
                            kws.append(part)
                            if len(kws) >= 20:  # Increased limit for logo-heavy motorsport images
                                break
                
                # Add category tags for car images
                kws.extend(["Automotive", "Vehicle"])
            else:
                # Non-car image - use new non-car fields for keywords
                kws.extend(["Scene", "Landscape"])
                # Enrich non-car keywords from new helper fields if present
                try:
                    nc_primary = _clean_str(meta.get('Primary Subject', meta.get('NC_Primary', ''))).replace('N/A','').strip()
                    nc_setting = _clean_str(meta.get('Setting', meta.get('NC_Setting', ''))).replace('N/A','').strip()
                    nc_keyobj = _clean_str(meta.get('Key Objects', meta.get('NC_KeyObjects', ''))).strip()
                    def _tokens(s: str) -> list[str]:
                        import re
                        # Split on commas and non-word boundaries, keep words 2..30 chars
                        parts = re.split(r"[\s,/;|]+", s)
                        out = []
                        for p in parts:
                            t = p.strip().strip('"').strip("'")
                            if 2 <= len(t) <= 30 and t.lower() not in {'n/a','none','null'}:
                                out.append(t)
                        return out
                    for t in _tokens(nc_primary) + _tokens(nc_setting) + _tokens(nc_keyobj):
                        kws.append(t)
                    # Heuristic: if setting contains common landscape terms, ensure a matching tag
                    setting_l = nc_setting.lower()
                    scenic_map = {
                        'lake': 'Lake', 'mountain': 'Mountain', 'peak': 'Mountain', 'forest': 'Forest',
                        'river': 'River', 'water': 'Water', 'sea': 'Sea', 'ocean': 'Ocean', 'beach': 'Beach',
                        'desert': 'Desert', 'valley': 'Valley', 'sky': 'Sky', 'sunrise': 'Sunrise', 'sunset': 'Sunset',
                    }
                    for kword, tag in scenic_map.items():
                        if kword in setting_l:
                            kws.append(tag)
                except Exception:
                    pass
            
            # Clean and dedupe keywords - more aggressive deduplication
            try:
                _seen = set()
                _cleaned = []
                for kw in kws:
                    s = str(kw).strip().strip('"').strip("'")
                    if not s or len(s) < 2 or len(s) > 30:
                        continue
                    if s.isdigit():
                        continue
                    # Skip generic terms that add no value
                    if s.lower() in {'car', 'vehicle', 'automotive', 'car photo'}:
                        continue
                    key_ci = s.casefold()
                    if key_ci not in _seen:
                        _cleaned.append(s)
                        _seen.add(key_ci)
                    if len(_cleaned) >= 15:  # Reduced cap for cleaner results
                        break
                
                # Add essential category tags at the end (only if not already present)
                # Only add Automotive/Vehicle for car images
                essential_tags = ["Automotive", "Vehicle"] if (not is_non_car and (make or model)) else []
                for tag in essential_tags:
                    if tag.casefold() not in _seen:
                        _cleaned.append(tag)
                        if len(_cleaned) >= 20:
                            break
                
                kws = _cleaned
                # If car detected, ensure scene-only tags are not present
                if not is_non_car and (make or model):
                    kws = [k for k in kws if k.lower() not in {'scene', 'landscape'}]
                # Final de-duplication preserving order (defensive)
                _seen2 = set()
                _final = []
                for k in kws:
                    kc = str(k).casefold()
                    if kc in _seen2:
                        continue
                    _seen2.add(kc)
                    _final.append(k)
                kws = _final
            except Exception:
                pass
            iptc_args = []
            # First clear all existing keywords
            iptc_args.append("-IPTC:Keywords=")
            # Then add each new keyword (without + to avoid appending)
            for kw in kws:
                iptc_args.append(f"-IPTC:Keywords={kw}")
            desc_parts = []
            title_parts = [p for p in [make, model] if p]
            
            # Extract race metadata fields (Team, Driver, Session, Car, Event, Heat, Race Track, Event Name)
            team = _clean_str(meta.get('Team', ''))
            driver = _clean_str(meta.get('Driver', ''))
            session = _clean_str(meta.get('Session', ''))
            race_car = _clean_str(meta.get('Car', ''))  # Race-specific car field (from PDF)
            event = _clean_str(meta.get('Event', ''))
            heat = _clean_str(meta.get('Heat', ''))
            race_track = _clean_str(meta.get('Race Track', ''))
            event_name = _clean_str(meta.get('Event Name', ''))
            
            # Only add car-specific description parts if this is not a non-car image
            if not is_non_car:
                if title_parts: desc_parts.append(f"Car: {' '.join(title_parts)}")
                if color: desc_parts.append(f"Color: {color}")
                if plate: desc_parts.append(f"License: {plate}")
                if logos: desc_parts.append(f"Logos: {str(logos)[:120]}")
                # Add race metadata to description if present
                if race_track: desc_parts.append(f"Race Track: {race_track}")
                if event_name: desc_parts.append(f"Event Name: {event_name}")
                if race_car: desc_parts.append(f"Race Car: {race_car}")
                if team: desc_parts.append(f"Team: {team}")
                if driver: desc_parts.append(f"Driver: {driver}")
                if event: desc_parts.append(f"Event: {event}")
                if heat: desc_parts.append(f"Heat: {heat}")
                if session: desc_parts.append(f"Session: {session}")
            else:
                # For non-car images, add NC_* field descriptions
                try:
                    nc_primary = _clean_str(meta.get('NC_Primary', '')).replace('N/A','').strip()
                    nc_setting = _clean_str(meta.get('NC_Setting', '')).replace('N/A','').strip()
                    nc_keyobj = _clean_str(meta.get('NC_KeyObjects', '')).strip()
                    if nc_primary: desc_parts.append(f"Subject: {nc_primary}")
                    if nc_setting: desc_parts.append(f"Setting: {nc_setting}")
                    if nc_keyobj: desc_parts.append(f"Objects: {nc_keyobj[:120]}")
                except Exception:
                    pass
            
            # Support both legacy and new summary key
            summary = _clean_str(meta.get(SUMMARY_KEY, meta.get('AI-Interpretation Summary', '')))
            # For cars, use the full AI Summary as the description (narrative)
            # For non-cars, build from parts
            # Title/ObjectName: use Make+Model for cars; otherwise derive from non-car fields
            # If PDF Car data exists, use that as source of truth for title
            noncar = is_non_car or (not make and not model)
            if not is_non_car:
                # Check if PDF Car data should override Make/Model
                if race_car and race_car.strip():
                    # PDF is source of truth - use PDF car for title
                    title = race_car.strip()
                    _log(f"[METADATA-WRITE] Using PDF Car '{title}' as title (overriding AI Make/Model)")
                elif title_parts:
                    title = ' '.join(title_parts)
                else:
                    title = 'Vehicle'
            elif is_non_car:
                try:
                    nc_primary = _clean_str(meta.get('NC_Primary', '')).replace('N/A','').strip()
                    nc_setting = _clean_str(meta.get('NC_Setting', '')).replace('N/A','').strip()
                except Exception:
                    nc_primary, nc_setting = '', ''
                title = nc_primary or nc_setting or 'Scene'
            elif noncar:
                try:
                    nc_primary = _clean_str(meta.get('NC_Primary', '')).replace('N/A','').strip()
                    nc_setting = _clean_str(meta.get('NC_Setting', '')).replace('N/A','').strip()
                except Exception:
                    nc_primary, nc_setting = '', ''
                title = nc_primary or nc_setting or 'Scene'
            elif summary:
                _s = summary.strip()
                _words = _s.split()
                title = ' '.join(_words[:10]).strip().rstrip('.') or 'Scene'
            else:
                title = 'Scene'
            
            # Description: For cars, use full AI Summary (narrative) + race metadata; for non-cars, use structured parts
            if not is_non_car and summary:
                # Combine summary with race metadata if present
                if desc_parts:
                    # Filter out basic car info (Make/Model) from desc_parts since summary already has it
                    race_metadata_parts = [p for p in desc_parts if not p.startswith('Car:') and not p.startswith('Color:') and not p.startswith('License:') and not p.startswith('Logos:')]
                    if race_metadata_parts:
                        description = f"{summary} | {' | '.join(race_metadata_parts)}"
                    else:
                        description = summary
                else:
                    description = summary  # Full narrative description
            else:
                description = ' - '.join(desc_parts) if desc_parts else summary
            
            # Generate Headline for both car and non-car images
            # If session is present (race metadata), use it as headline; otherwise use existing logic
            if session:
                headline = session  # Session takes priority for race images
            else:
                try:
                    if is_non_car:
                        # Non-car: combine primary and setting if available
                        try:
                            nc_primary = _clean_str(meta.get('NC_Primary', '')).replace('N/A','').strip()
                            nc_setting = _clean_str(meta.get('NC_Setting', '')).replace('N/A','').strip()
                        except Exception:
                            nc_primary, nc_setting = '', ''
                        head_src = ' '.join([p for p in [nc_primary, nc_setting] if p]).strip()
                        if not head_src and summary:
                            _s = summary.strip(); _words = _s.split(); head_src = ' '.join(_words[:12]).strip().rstrip('.')
                        headline = head_src or 'Scene'
                    elif make and model:
                        # Car image: use narrative excerpt from AI Summary if available, else make+model
                        if summary:
                            # Take first sentence or first 100 chars from summary for headline
                            first_sentence = summary.split('.')[0].strip()
                            headline = first_sentence[:100] if len(first_sentence) <= 100 else first_sentence[:97] + '...'
                        else:
                            headline = f"{make} {model}"
                    elif make:
                        # Car image with only make
                        headline = f"{make} Vehicle"
                    elif noncar:
                        # Fallback non-car: combine primary and setting if available
                        try:
                            nc_primary = _clean_str(meta.get('NC_Primary', '')).replace('N/A','').strip()
                            nc_setting = _clean_str(meta.get('NC_Setting', '')).replace('N/A','').strip()
                        except Exception:
                            nc_primary, nc_setting = '', ''
                        head_src = ' '.join([p for p in [nc_primary, nc_setting] if p]).strip()
                        if not head_src and summary:
                            _s = summary.strip(); _words = _s.split(); head_src = ' '.join(_words[:12]).strip().rstrip('.')
                        headline = head_src or 'Scene'
                    elif summary:
                        # Fallback for other non-car cases
                        _s = summary.strip(); _words = _s.split(); headline = ' '.join(_words[:12]).strip().rstrip('.') or 'Scene'
                    else:
                        headline = title
                except Exception:
                    headline = title or 'Scene'
                
                # Ensure headline is reasonable length (but don't truncate session if it's longer)
                if not session:  # Only truncate if not using session
                    headline = headline[:64]
                iptc_args.append(f"-IPTC:Headline={headline}")

            # Include Writer-Editor as requested (principled IPTC)
            writer_editor = f'www.tagomatic.co.uk {APP_VERSION}'
            
            # Clear existing metadata fields
            clear_fields = [
                '-IPTC:Keywords=', '-IPTC:Caption-Abstract=', '-IPTC:ObjectName=', 
                '-IPTC:Writer-Editor=', '-IPTC:Headline=',
                '-XMP-dc:Subject=', '-XMP-dc:Description=', '-XMP-dc:Title=',
                '-XPSubject=', '-XPComment=', '-XPTitle='
            ]
            
            # Build the command with explicit field mapping
            cmd2 = [exe, '-m', '-charset', 'UTF8', '-charset', 'iptc=UTF8', '-charset', 'filename=UTF8', 
                   '-overwrite_original', *clear_fields, *iptc_args,
                   f'-IPTC:Caption-Abstract={description}', 
                   f'-IPTC:ObjectName={title}', 
                   f'-IPTC:Writer-Editor={writer_editor}', 
                   f'-EXIF:ImageDescription={description}',
                   # Also write to XMP fields for better compatibility
                   f'-XMP-dc:Title={title}',
                   f'-XMP-dc:Description={description}']
            
            # Add race metadata fields to IPTC/XMP if present
            if team:
                cmd2.append(f'-IPTC:By-line={team}')  # IPTC By-line for team
                cmd2.append(f'-XMP-photoshop:AuthorsPosition={team}')  # XMP alternative
            if driver:
                cmd2.append(f'-IPTC:By-lineTitle={driver}')  # IPTC By-line Title for driver
                cmd2.append(f'-XMP-dc:Creator={driver}')  # XMP Creator for driver
            if session:
                cmd2.append(f'-IPTC:Headline={session}')  # IPTC Headline for session (overwrites previous headline if set)
                cmd2.append(f'-XMP-photoshop:Headline={session}')  # XMP Headline
            if race_car:
                cmd2.append(f'-IPTC:Category={race_car}')  # IPTC Category for race car
                cmd2.append(f'-XMP-photoshop:Category={race_car}')  # XMP Category
            if event:
                # Use IPTC:Location for event (or IPTC:Province-State as alternative)
                cmd2.append(f'-IPTC:Province-State={event}')  # IPTC Province/State for event
                cmd2.append(f'-XMP-photoshop:State={event}')  # XMP State for event
            if heat:
                # Use IPTC:Sub-location for heat
                cmd2.append(f'-IPTC:Sub-location=Heat {heat}')  # IPTC Sub-location for heat
                cmd2.append(f'-XMP-photoshop:City={heat}')  # XMP City for heat (alternative field)
            if race_track:
                # Use IPTC:Location for race track
                cmd2.append(f'-IPTC:Location={race_track}')  # IPTC Location for track
                cmd2.append(f'-XMP-photoshop:Country={race_track}')  # XMP Country for track (alternative)
            if event_name:
                # Use IPTC:Keywords for event name (add to keywords)
                cmd2.append(f'-IPTC:Keywords+={event_name}')  # Add event name to keywords
                cmd2.append(f'-XMP-dc:Subject+={event_name}')  # Add to XMP Subject
            
            cmd2.append(str(image_path))
            r2 = subprocess.run(cmd2, **kwargs)
            _log(f'rc2={r2.returncode}')
            if r2.stdout:
                _log(f'stdout2={r2.stdout.strip()}')
            if r2.stderr:
                _log(f'stderr2={r2.stderr.strip()}')
            if not ok and r2.returncode != 0 and r2.stderr and 'Temporary file already exists' in r2.stderr:
                try:
                    p = str(image_path)
                    candidates = [f"{p}_exiftool_tmp", f"{p.replace('\\', '/')}_exiftool_tmp", f"{p.replace('/', '\\')}_exiftool_tmp"]
                    for tpath in candidates:
                        try:
                            if os.path.exists(tpath):
                                os.remove(tpath)
                        except Exception:
                            pass
                    r2 = subprocess.run(cmd2, **kwargs)
                    _log(f'retry_rc2={r2.returncode}')
                    if r2.stdout:
                        _log(f'retry_stdout2={r2.stdout.strip()}')
                    if r2.stderr:
                        _log(f'retry_stderr2={r2.stderr.strip()}')
                except Exception:
                    pass
            ok = ok and (r2.returncode == 0)
        # Cleanup temp metadata file
        try:
            os.remove(tf_path)
        except Exception:
            pass
        return ok
    except Exception as e:
        try:
            with open(os.path.join(os.getcwd(), 'log.log'), 'a', encoding='utf-8') as lf:
                lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [EXIFWRITE] exception: {e}\n")
        except Exception:
            pass
        return False
class ResultItemWidget(QtWidgets.QWidget):
    rejected = Signal(dict)  # payload: { image_path, result, reason, comments, correct_make, correct_model, add_to_kb }
    def __init__(self, image_path: str, parsed: dict, raw: str):
        super().__init__()
        self.image_path = image_path
        self.parsed = parsed
        self.raw = raw
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        # Thumbnail
        thumb_label = QtWidgets.QLabel()
        thumb_label.setFixedSize(120, 90)
        try:
            img = Image.open(image_path)
            w, h = img.size
            scale = min(120 / w, 90 / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BOX)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            qimg = QtGui.QImage.fromData(buf.getvalue(), 'PNG')
            pix = QtGui.QPixmap.fromImage(qimg)
            thumb_label.setPixmap(pix)
        except Exception:
            pass
        layout.addWidget(thumb_label)

        # Inference mode indicator
        mode_label = QtWidgets.QLabel()
        mode_label.setFixedSize(80, 20)
        mode_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Determine inference mode from parsed data
        # Check for car fields first (most specific)
        car_fields = ['Make', 'Model', 'Color', 'Race Number', 'License Plate', 'Logos']
        has_car_fields = any(k in car_fields for k in (parsed or {}).keys())
        
        # Check for non-car fields (motorshow mode)
        has_nc_fields = any(k.startswith('nc_') for k in (parsed or {}).keys())
        
        if has_car_fields:
            mode_label.setText('Car Mode')
            mode_label.setStyleSheet('QLabel { background-color: #87ceeb; color: #1e3a8a; border-radius: 4px; font-size: 10px; font-weight: bold; }')
        elif has_nc_fields:
            mode_label.setText('Motorshow')
            mode_label.setStyleSheet('QLabel { background-color: #87ceeb; color: #1e3a8a; border-radius: 4px; font-size: 10px; font-weight: bold; }')
        else:
            mode_label.setText('Scene')
            mode_label.setStyleSheet('QLabel { background-color: #87ceeb; color: #1e3a8a; border-radius: 4px; font-size: 10px; font-weight: bold; }')
        
        layout.addWidget(mode_label)
        
        # Summary
        def _clean(v: str) -> str:
            try:
                s = str(v or '').strip()
                if s.startswith('**'):
                    s = s.lstrip('* ').strip()
                if s.endswith('**'):
                    s = s.rstrip('* ').strip()
                return s.strip('"').strip()
            except Exception:
                return str(v)

        text = QtWidgets.QLabel()
        rd = {str(k).lower(): v for k, v in (parsed or {}).items()}
        
        # Debug: Print what data we're working with
        print(f"[DEBUG] Display data keys: {list(rd.keys())}")
        print(f"[DEBUG] Looking for summary in keys: {[k for k in rd.keys() if 'summary' in k.lower() or 'ai' in k.lower()]}")
        
        # Check if this is a car or non-car image
        is_non_car = any(k.startswith('nc_') for k in (parsed or {}).keys())
        
        if is_non_car:
            # For non-car images, prioritize AI summary for rich content
            summary = rd.get('tagomatic ai summary') or rd.get('ai-interpretation summary') or rd.get('nc_descriptivesummary')
            print(f"[DEBUG] Found summary: {summary}")
            if summary:
                # Show more of the AI summary - it's the most valuable content
                display_text = str(summary)[:300] + "..." if len(str(summary)) > 300 else str(summary)
                text.setText(display_text)
            else:
                # Fallback to key non-car fields
                nc_parts = []
                if rd.get('nc_primary'): nc_parts.append(f"Subject: {_clean(rd.get('nc_primary'))}")
                if rd.get('nc_setting'): nc_parts.append(f"Setting: {_clean(rd.get('nc_setting'))}")
                if rd.get('nc_keyobjects'): 
                    key_objs = _clean(rd.get('nc_keyobjects'))
                    if key_objs and key_objs != 'N/A':
                        # Take first few key objects for display
                        key_list = [k.strip() for k in key_objs.split(',')[:3] if k.strip()]
                        if key_list:
                            nc_parts.append(f"Objects: {', '.join(key_list)}")
                
                if nc_parts:
                    text.setText(' | '.join(nc_parts))
                else:
                    # Show more useful fallback information
                    fallback_parts = []
                    if rd.get('nc_imagetype'): fallback_parts.append(f"Type: {_clean(rd.get('nc_imagetype'))}")
                    
                    if fallback_parts:
                        text.setText(' | '.join(fallback_parts))
                    else:
                        text.setText("✓ Analysis complete")
        else:
            # For car images, show car details; for non-car images, show AI summary
            if any(k.startswith('nc_') for k in rd.keys()):
                # Non-car image - show AI summary
                summary = rd.get('tagomatic ai summary') or rd.get('ai-interpretation summary') or rd.get('nc_descriptivesummary')
                if summary:
                    # Show more of the AI summary - it's the most valuable content
                    if len(summary) > 300:
                        summary = summary[:297] + "..."
                    text.setText(summary)
                else:
                    # Fallback to key non-car details
                    parts = []
                    if rd.get('nc_subject'): parts.append(f"Subject: {_clean(rd.get('nc_subject'))}")
                    if rd.get('nc_setting'): parts.append(f"Setting: {_clean(rd.get('nc_setting'))}")
                    if rd.get('nc_keyobjects'): 
                        key_objs = _clean(rd.get('nc_keyobjects'))
                        if key_objs and key_objs != 'N/A':
                            # Take first few key objects for display
                            key_list = [k.strip() for k in key_objs.split(',')[:3] if k.strip()]
                            if key_list:
                                parts.append(f"Objects: {', '.join(key_list)}")
                    
                    if parts:
                        text.setText(' | '.join(parts))
                    else:
                        # Show more useful fallback information
                        fallback_parts = []
                        if rd.get('nc_imagetype'): fallback_parts.append(f"Type: {_clean(rd.get('nc_imagetype'))}")
                        
                        if fallback_parts:
                            text.setText(' | '.join(fallback_parts))
                        else:
                            text.setText("✓ Analysis complete")
            else:
                # Car image - show car details + AI summary
                parts = []
                if rd.get('make'): parts.append(f"Make: {_clean(rd.get('make'))}")
                if rd.get('model'): parts.append(f"Model: {_clean(rd.get('model'))}")
                if rd.get('color'): parts.append(f"Color: {_clean(rd.get('color'))}")
                if rd.get('race number'): parts.append(f"# {_clean(rd.get('race number'))}")
                
                # Also try to show AI summary for car images
                summary = rd.get('tagomatic ai summary') or rd.get('ai-interpretation summary')
                if summary and len(str(summary).strip()) > 10:
                    # Show truncated AI summary for car images
                    if len(summary) > 200:
                        summary = summary[:197] + "..."
                    parts.append(summary)
                
                if parts:
                    text.setText(' | '.join(parts[:3]))
                else:
                    text.setText("✓ Analysis complete")
        
        text.setWordWrap(True)
        layout.addWidget(text, 1)

        # View button
        btn = QtWidgets.QPushButton('Details')
        btn.clicked.connect(self.show_details)
        layout.addWidget(btn)
        
        # Reject button next to Details (per-result)
        reject_btn = QtWidgets.QPushButton('Reject')
        reject_btn.setStyleSheet('QPushButton { background-color: #7a1f1f; color: #fff; }')
        reject_btn.setToolTip('Reject this result and provide feedback')
        reject_btn.clicked.connect(self.on_reject_clicked)
        layout.addWidget(reject_btn)

    def on_reject_clicked(self):
        """Reject this specific result item with feedback."""
        if not getattr(self, 'parsed', None):
            QtWidgets.QMessageBox.information(self, 'No Result', 'No result available to reject.')
            return
            
        # Create reject dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Reject Result - Provide Feedback')
        dlg.resize(500, 400)
        
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # Feedback options
        feedback_group = QtWidgets.QGroupBox('Rejection Reason')
        feedback_layout = QtWidgets.QVBoxLayout(feedback_group)
        
        reasons = [
            'Incorrect make/model',
            'Poor image quality',
            'No car visible',
            'Wrong vehicle type',
            'Incomplete result',
            'Other (specify below)'
        ]
        
        reason_buttons = []
        for reason in reasons:
            btn = QtWidgets.QRadioButton(reason)
            reason_buttons.append(btn)
            feedback_layout.addWidget(btn)
        
        # Select first reason by default
        if reason_buttons:
            reason_buttons[0].setChecked(True)
        
        layout.addWidget(feedback_group)
        
        # Knowledge Base section (for incorrect make/model)
        kb_group = QtWidgets.QGroupBox('Add to Knowledge Base (Optional)')
        kb_layout = QtWidgets.QVBoxLayout(kb_group)
        
        make_label = QtWidgets.QLabel('Correct Make:')
        make_input = QtWidgets.QLineEdit()
        make_input.setPlaceholderText('e.g., Ferrari')
        kb_layout.addWidget(make_label)
        kb_layout.addWidget(make_input)
        
        model_label = QtWidgets.QLabel('Correct Model:')
        model_input = QtWidgets.QLineEdit()
        model_input.setPlaceholderText('e.g., 488 GTB')
        kb_layout.addWidget(model_label)
        kb_layout.addWidget(model_input)
        
        add_to_kb_checkbox = QtWidgets.QCheckBox('Add to Knowledge Base (AI will extract features)')
        add_to_kb_checkbox.setEnabled(False)  # Enabled when Make/Model provided
        kb_layout.addWidget(add_to_kb_checkbox)
        
        # Enable checkbox when both fields have text
        def update_kb_checkbox():
            has_make = bool(make_input.text().strip())
            has_model = bool(model_input.text().strip())
            add_to_kb_checkbox.setEnabled(has_make and has_model)
        
        make_input.textChanged.connect(update_kb_checkbox)
        model_input.textChanged.connect(update_kb_checkbox)
        
        layout.addWidget(kb_group)
        
        # Additional comments
        comment_label = QtWidgets.QLabel('Additional Comments:')
        comment_text = QtWidgets.QTextEdit()
        comment_text.setMaximumHeight(100)
        layout.addWidget(comment_label)
        layout.addWidget(comment_text)
        
        # Increase dialog size to accommodate KB section
        dlg.resize(500, 550)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        reject_btn = QtWidgets.QPushButton('Confirm Rejection')
        reject_btn.setStyleSheet('QPushButton { background-color: #dc2626; color: #fff; }')
        cancel_btn = QtWidgets.QPushButton('Cancel')
        
        reject_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        
        btn_layout.addWidget(reject_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        # Show dialog
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            # Get selected reason
            selected_reason = 'Unknown'
            for i, btn in enumerate(reason_buttons):
                if btn.isChecked():
                    selected_reason = reasons[i]
                    break
            
            # Get comments
            comments = comment_text.toPlainText().strip()
            
            # Get KB fields
            correct_make = make_input.text().strip()
            correct_model = model_input.text().strip()
            add_to_kb = add_to_kb_checkbox.isChecked() if add_to_kb_checkbox.isEnabled() else False
            
            # Emit rejection signal with KB fields
            self.rejected.emit({
                'image_path': self.image_path,
                'result': self.parsed,
                'reason': selected_reason,
                'comments': comments,
                'correct_make': correct_make if add_to_kb else '',
                'correct_model': correct_model if add_to_kb else '',
                'add_to_kb': add_to_kb
            })

    def show_details(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Details - {os.path.basename(self.image_path)}")
        dlg.resize(1200, 800)
        outer = QtWidgets.QHBoxLayout(dlg)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # Left: Two organized metadata grids
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(16)
        
        # Style for better readability
        label_style = "QLabel { color: #dbeafe; background-color: #1e293b; padding: 4px 8px; border-radius: 4px; }"
        
        # Grid 1: EXIF UserComment (JSON) - What's stored in the image
        exif_group = QtWidgets.QGroupBox("EXIF UserComment (JSON Storage)")
        exif_group.setStyleSheet("QGroupBox { font-weight: bold; color: #93c5fd; } QGroupBox::title { color: #93c5fd; }")
        exif_layout = QtWidgets.QFormLayout(exif_group)
        exif_layout.setSpacing(8)
        exif_layout.setLabelAlignment(Qt.AlignRight)
        
        # Get the metadata that will be written
        meta = dict(self.parsed or {})
        is_non_car = any(k.startswith('NC_') for k in meta.keys())
        
        # Show relevant fields based on image type
        if is_non_car:
            # Non-car fields
            exif_layout.addRow("Image Type:", QtWidgets.QLabel(str(meta.get('NC_ImageType', 'N/A'))))
            exif_layout.addRow("Primary Subject:", QtWidgets.QLabel(str(meta.get('NC_Primary', 'N/A'))))
            exif_layout.addRow("Setting:", QtWidgets.QLabel(str(meta.get('NC_Setting', 'N/A'))))
            exif_layout.addRow("Key Objects:", QtWidgets.QLabel(str(meta.get('NC_KeyObjects', 'N/A'))))
            exif_layout.addRow("Descriptive Summary:", QtWidgets.QLabel(str(meta.get('NC_DescriptiveSummary', 'N/A'))))
        else:
            # Car fields
            # Check if PDF Car data exists (PDF is source of truth)
            race_car = str(meta.get('Car', '')).strip()
            if race_car:
                # PDF has car data - show it prominently and note it overrides AI
                exif_layout.addRow("Make:", QtWidgets.QLabel(str(meta.get('Make', 'N/A')) + " (from PDF)"))
                exif_layout.addRow("Model:", QtWidgets.QLabel(str(meta.get('Model', 'N/A')) + " (from PDF)"))
            else:
                # No PDF match - show AI identification
                exif_layout.addRow("Make:", QtWidgets.QLabel(str(meta.get('Make', 'N/A'))))
                exif_layout.addRow("Model:", QtWidgets.QLabel(str(meta.get('Model', 'N/A'))))
            exif_layout.addRow("Color:", QtWidgets.QLabel(str(meta.get('Color', 'N/A'))))
            exif_layout.addRow("Logos:", QtWidgets.QLabel(str(meta.get('Logos', 'N/A'))))
            exif_layout.addRow("Race Number:", QtWidgets.QLabel(str(meta.get('Race Number', 'N/A'))))
            exif_layout.addRow("License Plate:", QtWidgets.QLabel(str(meta.get('License Plate', 'N/A'))))
            
            # Race metadata fields (from PDF matching)
            driver = str(meta.get('Driver', '')).strip()
            session = str(meta.get('Session', '')).strip()
            team = str(meta.get('Team', '')).strip()
            event = str(meta.get('Event', '')).strip()
            heat = str(meta.get('Heat', '')).strip()
            race_track = str(meta.get('Race Track', '')).strip()
            event_name = str(meta.get('Event Name', '')).strip()
            
            if race_car or driver or session or team or event or heat or race_track or event_name:
                # Add separator or header for race metadata
                exif_layout.addRow("", QtWidgets.QLabel(""))  # Spacer
                race_header = QtWidgets.QLabel("Race Metadata (from PDF):")
                race_header.setStyleSheet("QLabel { color: #fbbf24; font-weight: bold; }")
                exif_layout.addRow(race_header, QtWidgets.QLabel(""))
                
                if race_track:
                    exif_layout.addRow("Race Track:", QtWidgets.QLabel(race_track))
                if event_name:
                    exif_layout.addRow("Event Name:", QtWidgets.QLabel(event_name))
                if race_car:
                    exif_layout.addRow("Car:", QtWidgets.QLabel(race_car))
                if driver:
                    exif_layout.addRow("Driver:", QtWidgets.QLabel(driver))
                if event:
                    exif_layout.addRow("Event:", QtWidgets.QLabel(event))
                if heat:
                    exif_layout.addRow("Heat:", QtWidgets.QLabel(heat))
                if session:
                    exif_layout.addRow("Session:", QtWidgets.QLabel(session))
                if team:
                    exif_layout.addRow("Team:", QtWidgets.QLabel(team))
        
        # Apply styling to all value labels in EXIF grid
        for i in range(exif_layout.rowCount()):
            item = exif_layout.itemAt(i, QtWidgets.QFormLayout.ItemRole.FieldRole)
            if item and item.widget():
                item.widget().setStyleSheet(label_style)
        
        # Common fields
        summary = meta.get('AI-Interpretation Summary', meta.get('TagoMatic AI Summary', ''))
        if summary:
            # Use QTextEdit for better multi-line display with scrolling
            summary_text = QtWidgets.QTextEdit(str(summary))
            summary_text.setReadOnly(True)
            summary_text.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
            summary_text.setMinimumHeight(120)  # Show more text at once
            summary_text.setMaximumHeight(300)  # Allow scrolling for very long summaries
            summary_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show scrollbar when needed
            summary_text.setStyleSheet(
                "QTextEdit { "
                "color: #dbeafe; "
                "background-color: #1e293b; "
                "padding: 8px; "
                "border-radius: 4px; "
                "border: 1px solid #334155; "
                "}"
            )
            exif_layout.addRow("AI Summary:", summary_text)
        
        # Provider info
        try:
            prov = None
            if isinstance(self.raw, dict) and self.raw.get('_provider'):
                prov = str(self.raw.get('_provider'))
            elif isinstance(self.raw, str) and self.raw.strip().startswith('{'):
                import json as _json
                try:
                    _obj = _json.loads(self.raw)
                    prov = _obj.get('_provider') if isinstance(_obj, dict) else None
                except Exception:
                    prov = None
            if prov:
                exif_layout.addRow("Provider:", QtWidgets.QLabel(prov))
        except Exception:
            pass
        
        left_layout.addWidget(exif_group)
        
        # Grid 2: IPTC Fields - What's visible in file managers
        iptc_group = QtWidgets.QGroupBox("IPTC Fields (File Manager Visible)")
        iptc_group.setStyleSheet("QGroupBox { font-weight: bold; color: #93c5fd; } QGroupBox::title { color: #93c5fd; }")
        iptc_layout = QtWidgets.QFormLayout(iptc_group)
        iptc_layout.setSpacing(8)
        iptc_layout.setLabelAlignment(Qt.AlignRight)
        
        # Generate what would be written to IPTC
        if is_non_car:
            # Non-car IPTC fields
            nc_primary = str(meta.get('NC_Primary', '')).replace('N/A','').strip()
            nc_setting = str(meta.get('NC_Setting', '')).replace('N/A','').strip()
            nc_keyobj = str(meta.get('NC_KeyObjects', '')).strip()
            
            title = nc_primary or nc_setting or 'Scene'
            headline = ' '.join([p for p in [nc_primary, nc_setting] if p]).strip() or 'Scene'
            description = ' - '.join([f"Subject: {nc_primary}" if nc_primary else "", 
                                    f"Setting: {nc_setting}" if nc_setting else "",
                                    f"Objects: {nc_keyobj}" if nc_keyobj else ""])
            
            iptc_layout.addRow("Title:", QtWidgets.QLabel(title))
            iptc_layout.addRow("Headline:", QtWidgets.QLabel(headline))
            desc_label = QtWidgets.QLabel(description[:100] + "..." if len(description) > 100 else description)
            desc_label.setWordWrap(True)
            iptc_layout.addRow("Description:", desc_label)
            iptc_layout.addRow("Key Objects:", QtWidgets.QLabel(nc_keyobj if nc_keyobj else "N/A"))
            
            # Generate intelligent keywords based on content
            keywords_parts = []
            if nc_primary and nc_primary != 'N/A':
                keywords_parts.append(nc_primary)
            if nc_setting and nc_setting != 'N/A':
                keywords_parts.append(nc_setting)
            if nc_keyobj and nc_keyobj != 'N/A':
                # Extract key terms from key objects
                key_terms = [term.strip() for term in nc_keyobj.split(',') if term.strip()]
                keywords_parts.extend(key_terms[:3])  # Limit to first 3 key terms
            
            # Add category based on content
            if any(word.lower() in ['church', 'building', 'interior', 'architecture'] for word in keywords_parts):
                keywords_parts.append('Architecture')
            elif any(word.lower() in ['landscape', 'nature', 'outdoor', 'scenery'] for word in keywords_parts):
                keywords_parts.append('Landscape')
            elif any(word.lower() in ['portrait', 'person', 'people', 'child'] for word in keywords_parts):
                keywords_parts.append('Portrait')
            else:
                keywords_parts.append('Scene')
            
            keywords = ', '.join(keywords_parts) if keywords_parts else 'Scene'
            iptc_layout.addRow("Keywords:", QtWidgets.QLabel(keywords))
        else:
            # Car IPTC fields
            make = str(meta.get('Make', '')).strip()
            model = str(meta.get('Model', '')).strip()
            color = str(meta.get('Color', '')).strip()
            race_number = str(meta.get('Race Number', '')).strip()
            plate = str(meta.get('License Plate', '')).strip()
            logos = str(meta.get('Logos', '')).strip()
            
            # Race metadata fields (from PDF matching)
            race_car = str(meta.get('Car', '')).strip()
            driver = str(meta.get('Driver', '')).strip()
            session = str(meta.get('Session', '')).strip()
            team = str(meta.get('Team', '')).strip()
            event = str(meta.get('Event', '')).strip()
            heat = str(meta.get('Heat', '')).strip()
            race_track = str(meta.get('Race Track', '')).strip()
            event_name = str(meta.get('Event Name', '')).strip()
            
            # If PDF Car data exists, use that as source of truth for title (PDF overrides AI)
            if race_car and race_car.strip():
                title = race_car.strip()  # PDF is source of truth
            else:
                title = ' '.join([p for p in [make, model] if p]) or 'Vehicle'
            # Use Session as headline if available (race metadata takes priority)
            headline = session if session else title
            description = ' - '.join([f"Car: {title}" if title != 'Vehicle' else "",
                                    f"Color: {color}" if color else "",
                                    f"Race Number: {race_number}" if race_number else "",
                                    f"License: {plate}" if plate else "",
                                    f"Logos: {logos}" if logos else ""])
            
            # Add race metadata to description if present
            if race_car or driver or session or team or event or heat or race_track or event_name:
                race_parts = []
                if race_track: race_parts.append(f"Race Track: {race_track}")
                if event_name: race_parts.append(f"Event Name: {event_name}")
                if race_car: race_parts.append(f"Race Car: {race_car}")
                if driver: race_parts.append(f"Driver: {driver}")
                if event: race_parts.append(f"Event: {event}")
                if heat: race_parts.append(f"Heat: {heat}")
                if session: race_parts.append(f"Session: {session}")
                if team: race_parts.append(f"Team: {team}")
                if race_parts:
                    description = f"{description} | {' | '.join(race_parts)}" if description else ' | '.join(race_parts)
            
            iptc_layout.addRow("Title:", QtWidgets.QLabel(title))
            iptc_layout.addRow("Headline:", QtWidgets.QLabel(headline))
            desc_label = QtWidgets.QLabel(description[:100] + "..." if len(description) > 100 else description)
            desc_label.setWordWrap(True)
            iptc_layout.addRow("Description:", desc_label)
            # Add Key Objects for car images (from logos, plate, race number, etc.)
            car_objects = []
            if logos: car_objects.append(f"Logos: {logos}")
            if race_number: car_objects.append(f"Race Number: {race_number}")
            if plate: car_objects.append(f"License: {plate}")
            key_objects = " - ".join(car_objects) if car_objects else "N/A"
            iptc_layout.addRow("Key Objects:", QtWidgets.QLabel(key_objects))
            iptc_layout.addRow("Keywords:", QtWidgets.QLabel(f"{make}, {model}, {color}, Race #{race_number}" if make and model and color and race_number else f"{make}, {model}, {color}" if make and model and color else "Automotive, Vehicle"))
        
        left_layout.addWidget(iptc_group)
        
        # Apply styling to all value labels in IPTC grid
        for i in range(iptc_layout.rowCount()):
            item = iptc_layout.itemAt(i, QtWidgets.QFormLayout.ItemRole.FieldRole)
            if item and item.widget():
                item.widget().setStyleSheet(label_style)
                # Enable word wrapping for longer text fields
                if isinstance(item.widget(), QtWidgets.QLabel):
                    item.widget().setWordWrap(True)
        
        # Add some stretch to push grids to top
        left_layout.addStretch(1)
        
        # Right: large preview
        preview = QtWidgets.QLabel()
        preview.setAlignment(Qt.AlignCenter)
        preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        preview.setScaledContents(True)
        try:
            img = Image.open(self.image_path)
            w, h = img.size
            target_w, target_h = 1200, 900
            scale = min(target_w/max(1,w), target_h/max(1,h), 1.0)
            img = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.Resampling.BOX)
            buf = io.BytesIO(); img.save(buf, format='PNG')
            qimg = QtGui.QImage.fromData(buf.getvalue(), 'PNG')
            pix = QtGui.QPixmap.fromImage(qimg)
            preview.setPixmap(pix)
        except Exception:
            preview.setText('Preview unavailable')

        splitter.addWidget(left_widget)
        splitter.addWidget(preview)
        try:
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
            splitter.setSizes([500, 700])
        except Exception:
            pass
        dlg.exec()


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, logo_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'About • Pistonspy : TagOmatic {APP_VERSION}')
        self.resize(720, 560)
        layout = QtWidgets.QVBoxLayout(self)

        # Logo
        logo = QtWidgets.QLabel()
        logo.setAlignment(Qt.AlignCenter)
        try:
            pix = QtGui.QPixmap(logo_path)
            if not pix.isNull():
                logo.setPixmap(pix.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            logo.setText('Pistonspy : TagOmatic')
        layout.addWidget(logo)

        # Rich text body
        body = QtWidgets.QTextBrowser()
        body.setOpenExternalLinks(True)
        body.setReadOnly(True)
        body.setStyleSheet('QTextBrowser { background-color: transparent; border: none; }')
        body.setHtml(
            (
                '<div style="font-family:Segoe UI, Roboto, Arial; color:#dbeafe;">'
                f'<h2 style="margin:0 0 8px 0; color:#93c5fd;">Pistonspy : TagOmatic <span style="font-size:14px;color:#bfdbfe;opacity:.85;margin-left:8px;">{APP_VERSION}</span></h2>'
                '<p style="margin:0 0 10px 0; color:#9cc3ff;">Local LLM‑powered car photo tagging</p>'
                '<p>This project began as a way to finally bring order to ~100k personal photos from a career break '
                'in the <b>Germany Eifel Region</b> (2009), where I fell in love with motorsport photography and prototype spy shots. '
                'For years I was too lazy to tag consistently, so images weren\'t truly searchable or editable. '
                'Pistonspy : TagOmatic fixes that by enriching each image with precise, standards‑compliant tags.</p>'
                '<p>It runs entirely on your machine using <b>Ollama</b> and two vision/LLM models for accuracy and speed:'
                ' <code>Gemma3:12b</code> and <code>Qwen2.5vl:7b</code>.</p>'
                '<ul>'
                '<li>Parses <b>Make</b>, <b>Model</b>, <b>Color</b>, <b>Logos</b>, <b>License Plate</b>, and a concise summary</li>'
                '<li>Writes standards‑compliant <b>IPTC</b> and EXIF metadata so your tags work across most galleries</li>'
                '<li>Privacy by default: inference is local; your images never leave your machine</li>'
                '</ul>'
                '<p>Connect with me on LinkedIn: '
                '<a href="https://www.linkedin.com/in/tonydewhurst/" style="color:#93c5fd;">'
                'https://www.linkedin.com/in/tonydewhurst/</a></p>'

                '<h3 style="color:#93c5fd;">Changelog v4.2</h3>'
                '<ul>'
                '<li><b>Three-Tier Knowledge Base System</b> - Built-in, Approved, and User KBs with priority merging (User > Approved > Built-in)</li>'
                '<li><b>Approved Knowledge Base</b> - Developer can bundle approved entries without overwriting user KB</li>'
                '<li><b>Knowledge Base Builder on Splash Screen</b> - Add entries to KB directly from batch review screen</li>'
                '<li><b>Enhanced KB Hint System</b> - Improved AI prompts with better model distinction for similar vehicles</li>'
                '<li><b>Race Metadata Tagging</b> - Premium feature for tagging images with Car/Team/Driver/Session from timing PDFs</li>'
                '<li><b>Multi-Format PDF Parser</b> - Auto-detects and supports both TSL and S.M.A.R.T timing PDF formats</li>'
                '<li><b>Batch Results Splash Screen</b> - Review all processed images with pagination, preview, and accept/reject</li>'
                '<li><b>Improved Validation System</b> - Better fuzzy matching and auto-retry with KB context</li>'
                '<li><b>User KB Management</b> - View, edit, and delete custom knowledge base entries</li>'
                '<li><b>Encrypted KB Export</b> - Secure export of user KB for submission to developer</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd;">Changelog v4.0</h3>'
                '<ul>'
                '<li><b>ONNX Runtime Vehicle Detection</b> - Replaced YOLO with thread-safe ONNX Runtime for intelligent vehicle detection and cropping</li>'
                '<li><b>Smart Dominant Vehicle Cropping</b> - Focuses on the largest vehicle with full + left + right detail crops (3 crops per vehicle)</li>'
                '<li><b>Enhanced Exotic Car Recognition</b> - Improved identification for Pagani Zonda/Huayra, TVR Sagaris, Koenigsegg, and other rare supercars</li>'
                '<li><b>Better Ferrari Model Distinction</b> - Enhanced prompts to distinguish 430 Scuderia vs Enzo, FF vs 488 GTB, and other similar models</li>'
                '<li><b>OCR Badge Text Extraction</b> - Dedicated OCR pass to extract badge/model text before main analysis for improved accuracy</li>'
                '<li><b>Expanded AI Summary</b> - Increased from 200 to 500 characters with scrollable UI display</li>'
                '<li><b>Improved UI Experience</b> - Results appear at bottom with auto-scroll, non-ASCII character filtering, better metadata display</li>'
                '<li><b>Multi-Vehicle Scene Handling</b> - Enhanced detection and identification for motorsport scenes with multiple vehicles</li>'
                '</ul>'

                '<h3 style="color:#93c5fd;">Special Thanks</h3>'
                '<p>A huge thank you to <a href="https://www.flatoutphotography.com/" style="color:#93c5fd;"><b>John Stewart from Flat Out Photography</b></a> '
                'for providing invaluable feedback and suggestions that helped shape this release. '
                'John\'s expertise in motorsport photography and his insights have been instrumental in improving '
                'our multi-vehicle detection and motorsport scene analysis capabilities.</p>'

                '<h3 style="color:#93c5fd;">Licensing</h3>'
                '<p>This app uses Qt for Python (<code>PySide6</code>) which is licensed under '
                '<b>LGPLv3</b>. It is dynamically linked at runtime and shipped with the original '
                'unmodified Qt/PySide6 libraries. Under LGPLv3 you may distribute a closed‑source '
                'application provided that users can relink against their own copies of the LGPL '
                'libraries (which this distribution allows by keeping the DLLs as separate files). '
                'If you modify those libraries, you must publish your changes under LGPLv3.</p>'
                '<ul>'
                '<li>Qt for Python (PySide6, shiboken6): LGPLv3</li>'
                '<li>Pillow: PIL license</li>'
                '<li>ExifTool: Artistic License 2.0</li>'
                '<li>Other bundled packages retain their respective licenses</li>'
                '</ul>'
                '</div>'
            )
        )
        layout.addWidget(body)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Help • Pistonspy : TagOmatic {APP_VERSION}')
        self.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(self)
        
        # Rich text body
        body = QtWidgets.QTextBrowser()
        body.setOpenExternalLinks(True)
        body.setReadOnly(True)
        body.setStyleSheet('QTextBrowser { background-color: transparent; border: none; }')
        body.setHtml(
            (
                '<div style="font-family:Segoe UI, Roboto, Arial; color:#dbeafe; line-height:1.6;">'
                
                '<h2 style="margin:0 0 8px 0; color:#93c5fd;">Usage Guide</h2>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Getting Started</h3>'
                '<p><b>1. Install Ollama and Models</b></p>'
                '<ul>'
                '<li>Download and install <a href="https://ollama.ai/" style="color:#93c5fd;">Ollama</a> from ollama.ai</li>'
                '<li>Open a terminal/command prompt and run: <code style="background:#1e293b; padding:2px 6px; border-radius:3px;">ollama pull qwen2.5vl:7b</code></li>'
                '<li>Then run: <code style="background:#1e293b; padding:2px 6px; border-radius:3px;">ollama pull gemma3:12b</code></li>'
                '<li>Wait for both models to download completely</li>'
                '</ul>'
                
                '<p><b>2. Configure Settings</b></p>'
                '<ul>'
                '<li>Select your preferred model from the <b>Model</b> dropdown (default: qwen2.5vl:32b-q4_K_M)</li>'
                '<li>Adjust <b>Max Concurrent</b> to control how many images process simultaneously (1-4)</li>'
                '<li>Enable <b>Recursive Scan</b> to process subfolders when selecting folders</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Processing Images</h3>'
                
                '<p><b>Single Image Processing</b></p>'
                '<ul>'
                '<li>Click <b>Process Image</b> button</li>'
                '<li>Select a single image file from the file dialog</li>'
                '<li>The image will be analyzed and tagged automatically</li>'
                '<li>Results appear in the <b>Completed Items</b> list at the bottom</li>'
                '<li>Metadata is written directly to the image file (IPTC and EXIF)</li>'
                '</ul>'
                
                '<p><b>Batch Folder Processing</b></p>'
                '<ul>'
                '<li>Click <b>Process Folders</b> button</li>'
                '<li>Select one or more folders containing images</li>'
                '<li>If <b>Recursive Scan</b> is enabled, subfolders are included automatically</li>'
                '<li>Processing happens in batches with progress tracking</li>'
                '<li>When batch completes, a review screen appears showing all results</li>'
                '<li>You can accept, reject, or retry individual images from the review screen</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Race Metadata Tagging (Premium Feature)</h3>'
                '<p><b>Event-PDF-Import</b> allows you to tag images with race-specific metadata from timing PDFs.</p>'
                '<ol>'
                '<li>Click the <b>Event-PDF-Import</b> button</li>'
                '<li>Select a timing PDF file (supports TSL and S.M.A.R.T formats)</li>'
                '<li>The button turns green with a checkmark when PDF is loaded successfully</li>'
                '<li>Now when you process images, they will be tagged with:</li>'
                '<ul>'
                '<li>Car Number</li>'
                '<li>Team Name</li>'
                '<li>Driver Name</li>'
                '<li>Session Type (Practice, Qualifying, Race, etc.)</li>'
                '<li>Track Name</li>'
                '<li>Event Name</li>'
                '</ul>'
                '<li>Enable <b>Tag by Race Number/Session Only</b> to skip full AI identification and use only PDF data (faster processing)</li>'
                '</ol>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Knowledge Base Management</h3>'
                
                '<p><b>Three-Tier System</b></p>'
                '<ul>'
                '<li><b>Built-in KB:</b> Pre-loaded with common vehicle information (lowest priority)</li>'
                '<li><b>Approved KB:</b> Developer-provided entries (medium priority)</li>'
                '<li><b>User KB:</b> Your custom entries (highest priority, never overwritten)</li>'
                '</ul>'
                
                '<p><b>Adding Entries to Knowledge Base</b></p>'
                '<ul>'
                '<li>During batch review, click <b>Reject</b> on an incorrectly identified image</li>'
                '<li>Fill in the correct Make and Model</li>'
                '<li>Check <b>Add to Knowledge Base</b> checkbox</li>'
                '<li>Optionally add reference images to help future identification</li>'
                '<li>Click <b>Submit</b> - the entry is saved to your User KB</li>'
                '</ul>'
                
                '<p><b>Managing Knowledge Base</b></p>'
                '<ul>'
                '<li>Click <b>KB Manager</b> to view all User KB entries</li>'
                '<li>Double-click an entry to preview details</li>'
                '<li>Select an entry and click <b>Delete Selected</b> to remove it</li>'
                '<li>Use <b>Export KB</b> from the Menu to create an encrypted ZIP for submission</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Controls and Settings</h3>'
                
                '<p><b>Action Buttons</b></p>'
                '<ul>'
                '<li><b>Process Image:</b> Tag a single image file</li>'
                '<li><b>Process Folders:</b> Tag all images in selected folder(s)</li>'
                '<li><b>Emergency Stop:</b> Immediately stop all processing and cancel in-flight tasks</li>'
                '<li><b>Pause:</b> Pause scheduling new items (in-flight tasks continue)</li>'
                '</ul>'
                
                '<p><b>Processing Controls</b></p>'
                '<ul>'
                '<li><b>Max Concurrent:</b> Number of simultaneous image analyses (1-4). Higher values = faster but more resource-intensive</li>'
                '<li><b>Recursive Scan:</b> When enabled, processes all subfolders when selecting a folder</li>'
                '<li><b>Motorshow Mode:</b> Auto-detects multi-vehicle/exhibition scenes and produces scene-first results</li>'
                '</ul>'
                
                '<p><b>Model Selection</b></p>'
                '<ul>'
                '<li>Choose from available Ollama models in the <b>Model</b> dropdown</li>'
                '<li>Recommended: <code style="background:#1e293b; padding:2px 6px; border-radius:3px;">qwen2.5vl:32b-q4_K_M</code> for best accuracy</li>'
                '<li>Smaller models (7b) are faster but may be less accurate</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Reviewing Results</h3>'
                
                '<p><b>Completed Items List</b></p>'
                '<ul>'
                '<li>All processed images appear in the <b>Completed Items</b> list at the bottom</li>'
                '<li>Click on any result to see detailed metadata</li>'
                '<li>History is preserved - all processed items remain in the list</li>'
                '<li>Each result shows: Make, Model, Color, Logos, License Plate, and AI Summary</li>'
                '</ul>'
                
                '<p><b>Batch Review Screen</b></p>'
                '<ul>'
                '<li>After batch processing completes, a review screen appears automatically</li>'
                '<li>Browse through results with pagination (50 items per page)</li>'
                '<li>Click an image to preview it</li>'
                '<li>Click <b>Accept</b> to confirm the result</li>'
                '<li>Click <b>Reject</b> to provide feedback and retry with corrections</li>'
                '<li>You can add entries to Knowledge Base directly from the review screen</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Tips and Best Practices</h3>'
                
                '<ul>'
                '<li><b>Start Small:</b> Test with a few images first to verify settings and model performance</li>'
                '<li><b>Use Knowledge Base:</b> Add entries for vehicles you photograph frequently to improve accuracy</li>'
                '<li><b>Reference Images:</b> When adding KB entries, include clear reference images for better future identification</li>'
                '<li><b>Batch Processing:</b> For large folders, use batch processing with appropriate Max Concurrent setting</li>'
                '<li><b>Race Tagging:</b> Load timing PDFs before processing race photos for automatic metadata enrichment</li>'
                '<li><b>Model Selection:</b> Larger models (32b) provide better accuracy but require more RAM and processing time</li>'
                '<li><b>Recursive Scan:</b> Enable when processing nested folder structures</li>'
                '<li><b>Motorshow Mode:</b> Enable for exhibition/show scenes with multiple vehicles</li>'
                '<li><b>Metadata Standards:</b> All tags are written in IPTC and EXIF formats for maximum compatibility</li>'
                '<li><b>Privacy:</b> All processing happens locally - your images never leave your machine</li>'
                '</ul>'
                
                '<h3 style="color:#93c5fd; margin-top:20px;">Troubleshooting</h3>'
                
                '<p><b>Model Not Found</b></p>'
                '<ul>'
                '<li>Ensure Ollama is running: <code style="background:#1e293b; padding:2px 6px; border-radius:3px;">ollama serve</code></li>'
                '<li>Verify models are installed: <code style="background:#1e293b; padding:2px 6px; border-radius:3px;">ollama list</code></li>'
                '<li>Pull missing models if needed</li>'
                '</ul>'
                
                '<p><b>Slow Processing</b></p>'
                '<ul>'
                '<li>Reduce <b>Max Concurrent</b> if system is overloaded</li>'
                '<li>Use smaller models (7b) for faster processing</li>'
                '<li>Close other resource-intensive applications</li>'
                '</ul>'
                
                '<p><b>Incorrect Identifications</b></p>'
                '<ul>'
                '<li>Add correct entries to Knowledge Base</li>'
                '<li>Use reference images when adding KB entries</li>'
                '<li>Try larger models for better accuracy</li>'
                '<li>Enable <b>Tag by Race Number/Session Only</b> when using PDF data for race photos</li>'
                '</ul>'
                
                '<p><b>PDF Import Issues</b></p>'
                '<ul>'
                '<li>Ensure PDF is a valid timing/results document</li>'
                '<li>Supported formats: TSL and S.M.A.R.T timing PDFs</li>'
                '<li>Check that PDF contains race data (car numbers, drivers, sessions)</li>'
                '</ul>'
                
                '</div>'
            )
        )
        layout.addWidget(body)
        
        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class RequirementsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Requirements • Pistonspy : TagOmatic {APP_VERSION}')
        self.resize(820, 620)
        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setReadOnly(True)
        text.setStyleSheet('QTextBrowser { background-color: transparent; border: none; }')
        text.setHtml(
            (
                '<div style="font-family:Segoe UI, Roboto, Arial; color:#dbeafe;">'
                '<h2 style="margin:0 0 8px 0; color:#93c5fd;">System Requirements</h2>'
                '<p>Pistonspy : TagOmatic runs locally on <b>Ollama</b>. Install Ollama first, then pull the models.</p>'

                '<h3 style="color:#93c5fd;">Install Ollama</h3>'
                '<p>Download for Windows/macOS/Linux from '
                '<a href="https://ollama.com" style="color:#93c5fd;">https://ollama.com</a>. '
                'After install, ensure the service is running (default: <code>http://localhost:11434</code>).</p>'

                '<h3 style="color:#93c5fd;">Recommended Hardware</h3>'
                '<ul>'
                '<li><b>GPU</b>: NVIDIA 8–12 GB VRAM recommended for smooth performance; CPU‑only works but is slower.</li>'
                '<li><b>RAM</b>: 16 GB or more recommended for large batches.</li>'
                '<li><b>Disk</b>: Models can be several GB; leave ~20 GB free for model storage and cache.</li>'
                '</ul>'

                '<h3 style="color:#93c5fd;">Pull Required Models</h3>'
                '<p>Open a terminal and run:</p>'
                '<pre style="background:#0f172a; border:1px solid #1f2937; padding:10px; border-radius:6px;">'
                'ollama pull qwen2.5vl:7b\n'
                'ollama pull gemma3:12b'
                '</pre>'
                '<p>After pulling, start the app and pick the models from the dropdowns.</p>'

                '<h3 style="color:#93c5fd;">Notes</h3>'
                '<ul>'
                '<li>First request per model incurs a warm‑up load; subsequent inferences are faster.</li>'
                '<li>Quantized variants can be selected in Ollama if you need lower memory usage.</li>'
                '<li>All tagging is written to IPTC/EXIF so galleries can index your results.</li>'
                '</ul>'
                '</div>'
            )
        )
        layout.addWidget(text)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

class BatchResultsSplashDialog(QtWidgets.QDialog):
    """Beautiful splash screen showing batch results with click-to-preview thumbnails."""
    rejected = Signal(dict)  # payload: { image_path, result, reason, comments, correct_make, correct_model, add_to_kb }
    accepted = Signal(dict)  # payload: { image_path, parsed, raw }
    
    def __init__(self, parent, results_data: list):
        super().__init__(parent)
        self.setWindowTitle('Batch Processing Complete - Review Results')
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        self.results_data = results_data
        self.rejected_paths = set()
        self.pending_accept = {}
        self.accepted_paths = set()
        self.items_per_page = 50
        self.current_page = 0
        
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        self.header = QtWidgets.QLabel()
        self.header.setStyleSheet('font-size: 18px; font-weight: bold; color: #1e40af; padding: 10px;')
        main_layout.addWidget(self.header)
        
        instructions = QtWidgets.QLabel('Click image to preview • Click "Reject" to provide feedback and retry • Click "Accept" to confirm')
        instructions.setStyleSheet('font-size: 12px; color: #64748b; padding: 5px;')
        main_layout.addWidget(instructions)
        
        pagination_layout = QtWidgets.QHBoxLayout()
        pagination_layout.addStretch()
        
        self.prev_btn = QtWidgets.QPushButton('◀ Previous')
        self.prev_btn.setStyleSheet('QPushButton { padding: 5px 15px; }')
        self.prev_btn.clicked.connect(self._prev_page)
        pagination_layout.addWidget(self.prev_btn)
        
        self.page_label = QtWidgets.QLabel('Page 1 of 1')
        self.page_label.setStyleSheet('padding: 5px 15px;')
        pagination_layout.addWidget(self.page_label)
        
        self.next_btn = QtWidgets.QPushButton('Next ▶')
        self.next_btn.setStyleSheet('QPushButton { padding: 5px 15px; }')
        self.next_btn.clicked.connect(self._next_page)
        pagination_layout.addWidget(self.next_btn)
        
        pagination_layout.addStretch()
        
        self.pagination_widget = QtWidgets.QWidget()
        self.pagination_widget.setLayout(pagination_layout)
        self.pagination_widget.setVisible(False)
        main_layout.addWidget(self.pagination_widget)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border: none; }')
        
        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        
        scroll.setWidget(self.grid_widget)
        main_layout.addWidget(scroll, 1)
        
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.setStyleSheet('QPushButton { padding: 8px 20px; font-size: 14px; }')
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)
        
        self._refresh_grid()
        self._update_header()
        self._update_pagination()
    
    def _update_header(self):
        if not hasattr(self, 'pagination_widget'):
            return
        total = len(self.results_data)
        rejected = len(self.rejected_paths)
        accepted = len(self.accepted_paths)
        pending = len(self.pending_accept)
        remaining = total - rejected - accepted - pending
        self.header.setText(f'Batch Complete: {remaining} items remaining ({total} total, {rejected} rejected, {accepted} accepted, {pending} pending retry)')
    
    def _update_pagination(self):
        if not hasattr(self, 'pagination_widget'):
            return
        total_items = len([r for r in self.results_data if r['image_path'] not in self.rejected_paths and r['image_path'] not in self.accepted_paths])
        total_pages = (total_items + self.items_per_page - 1) // self.items_per_page if total_items > 0 else 1
        self.pagination_widget.setVisible(total_pages > 1)
        self.page_label.setText(f'Page {self.current_page + 1} of {total_pages}')
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < total_pages - 1)
    
    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._refresh_grid()
            self._update_pagination()
    
    def _next_page(self):
        total_items = len([r for r in self.results_data if r['image_path'] not in self.rejected_paths and r['image_path'] not in self.accepted_paths])
        total_pages = (total_items + self.items_per_page - 1) // self.items_per_page if total_items > 0 else 1
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._refresh_grid()
            self._update_pagination()
    
    def _refresh_grid(self):
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        visible_results = [r for r in self.results_data if r['image_path'] not in self.rejected_paths and r['image_path'] not in self.accepted_paths]
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_results = visible_results[start_idx:end_idx]
        
        for idx, result in enumerate(page_results):
            widget = self._create_thumbnail_widget(result, idx)
            row = idx // 5
            col = idx % 5
            self.grid_layout.addWidget(widget, row, col)
    
    def _create_thumbnail_widget(self, result: dict, idx: int) -> QtWidgets.QWidget:
        image_path = result.get('image_path', '')
        parsed = result.get('parsed', {})
        raw = result.get('raw', '')
        
        container = QtWidgets.QWidget()
        container.setFixedSize(200, 280)
        container.setStyleSheet('QWidget { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; }')
        
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        
        thumb_label = QtWidgets.QLabel()
        thumb_label.setFixedSize(184, 138)
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setStyleSheet('QLabel { background-color: #0f172a; border: 1px solid #475569; border-radius: 4px; }')
        thumb_label.setScaledContents(False)
        
        try:
            img = Image.open(image_path)
            img.thumbnail((184, 138), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            qimg = QtGui.QImage.fromData(buf.getvalue(), 'PNG')
            pix = QtGui.QPixmap.fromImage(qimg)
            thumb_label.setPixmap(pix)
        except Exception:
            thumb_label.setText('No Image')
        
        thumb_label.mousePressEvent = lambda e, path=image_path, p=parsed, r=raw: self._show_preview(path, p, r)
        thumb_label.setCursor(Qt.PointingHandCursor)
        layout.addWidget(thumb_label)
        
        make = parsed.get('Make', 'Unknown')
        model = parsed.get('Model', 'Unknown')
        info_text = f"{make} {model}"[:30]
        info_label = QtWidgets.QLabel(info_text)
        info_label.setStyleSheet('QLabel { color: #cfe3ff; font-size: 11px; }')
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(5)
        
        accept_btn = QtWidgets.QPushButton('Accept')
        accept_btn.setStyleSheet('QPushButton { background-color: #16a34a; color: #fff; padding: 4px; font-size: 10px; }')
        accept_btn.clicked.connect(lambda checked, path=image_path, p=parsed, r=raw: self._accept_result(path, p, r))
        btn_layout.addWidget(accept_btn)
        
        reject_btn = QtWidgets.QPushButton('Reject')
        reject_btn.setStyleSheet('QPushButton { background-color: #dc2626; color: #fff; padding: 4px; font-size: 10px; }')
        reject_btn.clicked.connect(lambda checked, res=result: self._open_reject_dialog(res))
        btn_layout.addWidget(reject_btn)
        
        layout.addLayout(btn_layout)
        return container
    
    def _accept_result(self, image_path: str, parsed: dict, raw: str):
        self.pending_accept.pop(image_path, None)
        self.accepted_paths.add(image_path)
        self.accepted.emit({'image_path': image_path, 'parsed': parsed, 'raw': raw})
        self._refresh_grid()
        self._update_header()
        self._update_pagination()
    
    def add_retry_result(self, image_path: str, parsed: dict, raw: str):
        self.pending_accept[image_path] = {'parsed': parsed, 'raw': raw}
        self.rejected_paths.discard(image_path)
        for r in self.results_data:
            if r['image_path'] == image_path:
                r['parsed'] = parsed
                r['raw'] = raw
                break
        else:
            self.results_data.append({'image_path': image_path, 'parsed': parsed, 'raw': raw})
        self._refresh_grid()
        self._update_header()
        self._update_pagination()
    
    def _show_preview(self, image_path: str, parsed: dict, raw: str):
        preview_dlg = QtWidgets.QDialog(self)
        preview_dlg.setWindowTitle(f"Preview - {os.path.basename(image_path)}")
        preview_dlg.resize(1200, 800)
        layout = QtWidgets.QVBoxLayout(preview_dlg)
        
        img_label = QtWidgets.QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        try:
            img = Image.open(image_path)
            w, h = img.size
            if h > 700:
                scale = 700 / h
                new_w, new_h = int(w * scale), 700
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            qimg = QtGui.QImage.fromData(buf.getvalue(), 'PNG')
            pix = QtGui.QPixmap.fromImage(qimg)
            img_label.setPixmap(pix)
        except Exception:
            img_label.setText('Failed to load image')
        layout.addWidget(img_label)
        
        meta_text = QtWidgets.QTextEdit()
        meta_text.setReadOnly(True)
        meta = dict(parsed or {})
        meta_lines = [f"{key}: {value}" for key, value in meta.items()]
        meta_text.setText('\n'.join(meta_lines))
        layout.addWidget(meta_text)
        
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(preview_dlg.accept)
        layout.addWidget(close_btn)
        
        preview_dlg.exec()
    
    def _open_reject_dialog(self, result: dict):
        image_path = result.get('image_path', '')
        parsed = result.get('parsed', {})
        
        if not parsed:
            QtWidgets.QMessageBox.information(self, 'No Result', 'No result available to reject.')
            return
        
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Reject Result - Provide Feedback')
        dlg.resize(500, 550)
        
        layout = QtWidgets.QVBoxLayout(dlg)
        
        feedback_group = QtWidgets.QGroupBox('Rejection Reason')
        feedback_layout = QtWidgets.QVBoxLayout(feedback_group)
        
        reasons = ['Incorrect make/model', 'Poor image quality', 'No car visible', 'Wrong vehicle type', 'Incomplete result', 'Other (specify below)']
        reason_buttons = []
        for reason in reasons:
            btn = QtWidgets.QRadioButton(reason)
            reason_buttons.append(btn)
            feedback_layout.addWidget(btn)
        
        if reason_buttons:
            reason_buttons[0].setChecked(True)
        
        layout.addWidget(feedback_group)
        
        kb_group = QtWidgets.QGroupBox('Add to Knowledge Base (Optional)')
        kb_layout = QtWidgets.QVBoxLayout(kb_group)
        
        make_label = QtWidgets.QLabel('Correct Make:')
        make_input = QtWidgets.QLineEdit()
        make_input.setPlaceholderText('e.g., Ferrari')
        kb_layout.addWidget(make_label)
        kb_layout.addWidget(make_input)
        
        model_label = QtWidgets.QLabel('Correct Model:')
        model_input = QtWidgets.QLineEdit()
        model_input.setPlaceholderText('e.g., 488 GTB')
        kb_layout.addWidget(model_label)
        kb_layout.addWidget(model_input)
        
        add_to_kb_checkbox = QtWidgets.QCheckBox('Add to Knowledge Base (AI will extract features)')
        add_to_kb_checkbox.setEnabled(False)
        kb_layout.addWidget(add_to_kb_checkbox)
        
        def update_kb_checkbox():
            has_make = bool(make_input.text().strip())
            has_model = bool(model_input.text().strip())
            add_to_kb_checkbox.setEnabled(has_make and has_model)
        
        make_input.textChanged.connect(update_kb_checkbox)
        model_input.textChanged.connect(update_kb_checkbox)
        
        layout.addWidget(kb_group)
        
        comments_label = QtWidgets.QLabel('Additional Comments:')
        layout.addWidget(comments_label)
        comments_text = QtWidgets.QTextEdit()
        comments_text.setPlaceholderText('Provide specific feedback to improve the result...')
        comments_text.setMaximumHeight(100)
        layout.addWidget(comments_text)
        
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.clicked.connect(dlg.reject)
        button_layout.addWidget(cancel_btn)
        
        submit_btn = QtWidgets.QPushButton('Submit Rejection')
        submit_btn.setStyleSheet('QPushButton { background-color: #7a1f1f; color: #fff; }')
        submit_btn.clicked.connect(lambda: self._submit_rejection(
            dlg, image_path, parsed, 
            reasons[[btn.isChecked() for btn in reason_buttons].index(True)] if any(btn.isChecked() for btn in reason_buttons) else 'Unknown',
            comments_text.toPlainText(),
            make_input.text().strip(),
            model_input.text().strip(),
            add_to_kb_checkbox.isChecked() if add_to_kb_checkbox.isEnabled() else False
        ))
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)
        dlg.exec()
    
    def _submit_rejection(self, dlg, image_path: str, parsed: dict, reason: str, comments: str, correct_make: str = '', correct_model: str = '', add_to_kb: bool = False):
        if not reason:
            QtWidgets.QMessageBox.warning(dlg, 'No Reason', 'Please select a rejection reason.')
            return
        
        self.rejected_paths.add(image_path)
        self.pending_accept.pop(image_path, None)
        
        self.rejected.emit({
            'image_path': image_path,
            'result': parsed,
            'reason': reason,
            'comments': comments,
            'correct_make': correct_make if add_to_kb else '',
            'correct_model': correct_model if add_to_kb else '',
            'add_to_kb': add_to_kb
        })
        dlg.accept()
        
        self._refresh_grid()
        self._update_header()
        self._update_pagination()

class KnowledgeBaseManagerDialog(QtWidgets.QDialog):
    """Dialog to view and manage user knowledge base entries."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Knowledge Base Manager')
        self.resize(900, 600)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Info label
        info_label = QtWidgets.QLabel('User Knowledge Base Entries (highest priority - never overwritten)')
        info_label.setStyleSheet('QLabel { color: #93c5fd; font-weight: bold; padding: 8px; }')
        layout.addWidget(info_label)
        
        # List widget for entries
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self._preview_entry)
        # Ensure clicking an item sets it as current and selected
        def on_item_clicked(item):
            self.list_widget.setCurrentItem(item)
            item.setSelected(True)
        self.list_widget.itemClicked.connect(on_item_clicked)
        layout.addWidget(self.list_widget)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        refresh_btn = QtWidgets.QPushButton('Refresh')
        refresh_btn.clicked.connect(self._refresh_list)
        btn_layout.addWidget(refresh_btn)
        
        preview_btn = QtWidgets.QPushButton('Preview Selected')
        # Use lambda to ensure we don't pass button state
        preview_btn.clicked.connect(lambda: self._preview_entry())
        btn_layout.addWidget(preview_btn)
        
        delete_btn = QtWidgets.QPushButton('Delete Selected')
        delete_btn.setStyleSheet('QPushButton { background-color: #dc2626; color: #fff; }')
        delete_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(delete_btn)
        
        cleanup_btn = QtWidgets.QPushButton('Cleanup Unreferenced Images')
        cleanup_btn.setToolTip('Remove images from kb_references folder that are not referenced in any KB JSON file')
        cleanup_btn.clicked.connect(self._cleanup_images)
        btn_layout.addWidget(cleanup_btn)
        
        btn_layout.addStretch()
        
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Load entries
        self._refresh_list()
    
    def _refresh_list(self):
        """Reload and display KB entries."""
        self.list_widget.clear()
        
        try:
            kb = load_user_knowledge_base()
            
            if not kb:
                item = QtWidgets.QListWidgetItem('No entries in user knowledge base')
                item.setFlags(Qt.NoItemFlags)  # Disable selection
                self.list_widget.addItem(item)
                return
            
            for make, models in kb.items():
                if isinstance(models, list):
                    for entry in models:
                        if isinstance(entry, dict):
                            model = entry.get('model', 'Unknown')
                            cues = entry.get('distinguishing_cues', {})
                            ref_img = entry.get('reference_image', '')
                            
                            # Create display text
                            text = f"{make} {model}"
                            if ref_img:
                                text += f" [Reference: {os.path.basename(ref_img)}]"
                            
                            # Count non-empty cues
                            cue_count = sum(1 for v in cues.values() if v)
                            if cue_count > 0:
                                text += f" ({cue_count} distinguishing features)"
                            
                            item = QtWidgets.QListWidgetItem(text)
                            item.setData(Qt.UserRole, {'make': make, 'model': model, 'entry': entry})
                            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                            self.list_widget.addItem(item)
        except Exception as e:
            _log(f"Error refreshing KB list: {e}")
            import traceback
            _log(traceback.format_exc())
            item = QtWidgets.QListWidgetItem(f'Error loading KB: {e}')
            item.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(item)
    
    def _delete_selected(self):
        """Delete selected KB entry."""
        current = self.list_widget.currentItem()
        if not current:
            QtWidgets.QMessageBox.warning(self, 'No Selection', 'Please select an entry to delete.')
            return
        
        data = current.data(Qt.UserRole)
        if not data:
            return
        
        make = data.get('make')
        model = data.get('model')
        
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self,
            'Confirm Delete',
            f'Delete entry for {make} {model}?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        try:
            # Load KB
            kb_path = os.path.join(_app_base_dir(), 'user_knowledge_base.json')
            kb = load_user_knowledge_base(kb_path)
            
            if make in kb and isinstance(kb[make], list):
                # Remove entry
                kb[make] = [e for e in kb[make] if isinstance(e, dict) and e.get('model', '').lower() != model.lower()]
                
                # Remove make if empty
                if not kb[make]:
                    del kb[make]
                
                # Save
                with open(kb_path, 'w', encoding='utf-8') as f:
                    json.dump(kb, f, indent=2, ensure_ascii=False)
                
                # Clear cache
                global _user_kb_cache
                _user_kb_cache = None
                
                # Refresh list
                self._refresh_list()
                
                QtWidgets.QMessageBox.information(self, 'Deleted', f'Entry for {make} {model} deleted successfully.')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Error deleting entry: {e}')
            _log(f"Error deleting KB entry: {e}")
            import traceback
            _log(traceback.format_exc())
    
    def _cleanup_images(self):
        """Clean up unreferenced images from kb_references folder."""
        reply = QtWidgets.QMessageBox.question(
            self,
            'Confirm Cleanup',
            'This will remove all images from kb_references folder that are not referenced in any KB JSON file.\n\nContinue?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        try:
            deleted_count, deleted_files = cleanup_unreferenced_kb_images()
            
            if deleted_count > 0:
                files_list = '\n'.join(deleted_files[:10])  # Show first 10
                if len(deleted_files) > 10:
                    files_list += f'\n... and {len(deleted_files) - 10} more'
                
                QtWidgets.QMessageBox.information(
                    self,
                    'Cleanup Complete',
                    f'Removed {deleted_count} unreferenced image(s):\n\n{files_list}'
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    'Cleanup Complete',
                    'No unreferenced images found. All images in kb_references are referenced in KB JSON files.'
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Error during cleanup: {e}')
            _log(f"Error in KB cleanup: {e}")
            import traceback
            _log(traceback.format_exc())
    
    def _preview_entry(self, item=None):
        """Preview selected KB entry with image and details."""
        # Debug logging
        _log(f"[KB-PREVIEW] Called with item={item}, type={type(item)}")
        
        # If item is not a QListWidgetItem (e.g., False from button click), get it from selection
        if not isinstance(item, QtWidgets.QListWidgetItem):
            # Try multiple ways to get the selected item
            selected = self.list_widget.selectedItems()
            _log(f"[KB-PREVIEW] selectedItems() returned: {len(selected)} items")
            if selected:
                item = selected[0]
                _log(f"[KB-PREVIEW] Using first selected item: {item.text()}")
            else:
                item = self.list_widget.currentItem()
                _log(f"[KB-PREVIEW] currentItem() returned: {item.text() if item else 'None'}")
        
        if not item or not isinstance(item, QtWidgets.QListWidgetItem):
            _log(f"[KB-PREVIEW] No valid item found! item={item}, type={type(item)}")
            QtWidgets.QMessageBox.warning(self, 'No Selection', 'Please select an entry to preview.')
            return
        
        # Check if item is selectable (not a placeholder/error message)
        if not (item.flags() & Qt.ItemIsSelectable):
            _log(f"[KB-PREVIEW] Item not selectable: {item.text()}, flags={item.flags()}")
            QtWidgets.QMessageBox.warning(self, 'Invalid Selection', 'This item cannot be previewed.')
            return
        
        data = item.data(Qt.UserRole)
        _log(f"[KB-PREVIEW] Item data: {data is not None}, text: {item.text()}")
        if not data:
            # Debug: log what we got
            _log(f"[KB-PREVIEW] Item has no UserRole data. Text: {item.text()}, Flags: {item.flags()}")
            QtWidgets.QMessageBox.warning(self, 'Invalid Entry', 'Selected entry has no data. Please refresh the list.')
            return
        
        entry = data.get('entry', {})
        make = data.get('make', 'Unknown')
        model = data.get('model', 'Unknown')
        
        # Create preview dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f'KB Preview - {make} {model}')
        dlg.resize(1000, 700)
        
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # Splitter for image and details
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left: Reference image
        image_widget = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_widget)
        image_layout.setContentsMargins(8, 8, 8, 8)
        
        image_label = QtWidgets.QLabel('No reference image')
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet('QLabel { background-color: #1e293b; border: 1px solid #334155; border-radius: 4px; padding: 20px; color: #94a3b8; }')
        image_label.setMinimumSize(400, 300)
        image_label.setScaledContents(False)
        
        ref_img_path = entry.get('reference_image', '')
        if ref_img_path and os.path.exists(ref_img_path):
            try:
                from PIL import Image
                img = Image.open(ref_img_path)
                # Resize for display (max 500px width/height)
                img.thumbnail((500, 500), Image.Resampling.LANCZOS)
                
                # Convert to QPixmap
                import io
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                qimg = QtGui.QImage.fromData(buf.getvalue())
                pixmap = QtGui.QPixmap.fromImage(qimg)
                image_label.setPixmap(pixmap)
                image_label.setText('')
                image_label.setAlignment(Qt.AlignCenter)
            except Exception as e:
                image_label.setText(f'Error loading image:\n{str(e)}')
                _log(f"Error loading KB reference image {ref_img_path}: {e}")
        elif ref_img_path:
            image_label.setText(f'Image not found:\n{ref_img_path}')
        
        image_layout.addWidget(QtWidgets.QLabel('Reference Image:'))
        image_layout.addWidget(image_label)
        image_layout.addStretch()
        
        splitter.addWidget(image_widget)
        
        # Right: Entry details
        details_widget = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details_widget)
        details_layout.setContentsMargins(8, 8, 8, 8)
        
        # Scroll area for details
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border: 1px solid #334155; border-radius: 4px; }')
        
        details_content = QtWidgets.QWidget()
        details_content_layout = QtWidgets.QVBoxLayout(details_content)
        details_content_layout.setSpacing(12)
        
        # Make/Model
        make_model_label = QtWidgets.QLabel(f'<h2>{make} {model}</h2>')
        make_model_label.setStyleSheet('QLabel { color: #93c5fd; padding: 8px; }')
        details_content_layout.addWidget(make_model_label)
        
        # Distinguishing cues
        cues = entry.get('distinguishing_cues', {})
        if cues:
            cues_group = QtWidgets.QGroupBox('Distinguishing Features')
            cues_group.setStyleSheet('QGroupBox { font-weight: bold; color: #93c5fd; } QGroupBox::title { color: #93c5fd; }')
            cues_layout = QtWidgets.QFormLayout(cues_group)
            cues_layout.setSpacing(8)
            cues_layout.setLabelAlignment(Qt.AlignRight)
            
            # Display all cue fields that have values
            # First check for common predefined fields
            cue_fields = {
                'body_shape': 'Body Shape',
                'logo_location': 'Logo Location',
                'unique_grills': 'Unique Grills',
                'light_clusters': 'Light Clusters',
                'side_features': 'Side Features',
                'rear_features': 'Rear Features',
                'roof_features': 'Roof Features',
                'other_features': 'Other Features',
                'front': 'Front',
                'side': 'Side',
                'rear': 'Rear',
                'grille': 'Grille',
                'lights': 'Lights',
                'logo': 'Logo'
            }
            
            # Track which fields we've displayed
            displayed_keys = set()
            
            # Display predefined fields first
            for key, label in cue_fields.items():
                value = cues.get(key, '')
                if value:
                    label_widget = QtWidgets.QLabel(str(value))
                    label_widget.setWordWrap(True)
                    cues_layout.addRow(f'{label}:', label_widget)
                    displayed_keys.add(key)
            
            # Display any remaining cue fields that weren't in the predefined list
            for key, value in cues.items():
                if key not in displayed_keys and value:
                    # Format key name (e.g., "body_shape" -> "Body Shape")
                    label = key.replace('_', ' ').title()
                    label_widget = QtWidgets.QLabel(str(value))
                    label_widget.setWordWrap(True)
                    cues_layout.addRow(f'{label}:', label_widget)
            
            details_content_layout.addWidget(cues_group)
        
        # Additional metadata
        meta_group = QtWidgets.QGroupBox('Additional Information')
        meta_group.setStyleSheet('QGroupBox { font-weight: bold; color: #93c5fd; } QGroupBox::title { color: #93c5fd; }')
        meta_layout = QtWidgets.QFormLayout(meta_group)
        meta_layout.setSpacing(8)
        meta_layout.setLabelAlignment(Qt.AlignRight)
        
        # Show all other entry fields
        for key, value in entry.items():
            if key not in ['distinguishing_cues', 'reference_image', 'model']:
                if value:
                    meta_layout.addRow(f'{key.replace("_", " ").title()}:', QtWidgets.QLabel(str(value)))
        
        if ref_img_path:
            meta_layout.addRow('Reference Image Path:', QtWidgets.QLabel(ref_img_path))
        
        details_content_layout.addWidget(meta_group)
        details_content_layout.addStretch()
        
        scroll.setWidget(details_content)
        details_layout.addWidget(scroll)
        
        splitter.addWidget(details_widget)
        
        # Set splitter proportions (40% image, 60% details)
        splitter.setSizes([400, 600])
        
        # Close button
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(dlg.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dlg.exec()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Pistonspy : TagOmatic {APP_VERSION}')
        self.resize(1200, 800)
        
        # Initialize result tracking for reject functionality
        self.current_result = None
        # App/logo icon
        try:
            # Window/app icon
            app_icon_path = _resource_path('tagomatic_main.ico')
            app_icon = QtGui.QIcon(app_icon_path)
            if not app_icon.isNull():
                self.setWindowIcon(app_icon)
            # UI logo (status/header) can remain the blue logo
            self._app_logo_path = _resource_path('logo44.png')
        except Exception:
            pass

        self.client = ollama.Client(host='http://127.0.0.1:11434', timeout=300)  # Increased from 60 to 300 seconds
        self.model_name = 'qwen2.5vl:32b-q4_K_M'
        self.previous_model_name = None  # Track previous model for unloading
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self.threadpool.setMaxThreadCount(4)  # Set initial thread pool size
        self.max_concurrent = 2
        # Dedicated, throttled pool for metadata existence checks
        self.meta_pool = QtCore.QThreadPool()
        try:
            self.meta_pool.setMaxThreadCount(3)
        except Exception:
            pass
        self.inflight = set()
        self._active_tasks: dict[str, tuple['InferenceTask', 'TaskRunnable']] = {}
        self._active_meta: dict[str, tuple['MetaCheckTask', 'MetaRunnable']] = {}
        self.is_batch = False
        self.batch_total = 0
        self.batch_done = 0
        self.batch_failed = 0
        self.batch_skipped = 0
        # Concurrency ramp & timeouts
        self.dynamic_limit = 1  # Start at 1, will be updated when user changes max_concurrent
        self.success_streak = 0
        self.failure_streak = 0
        self.per_image_timeout_secs = 300  # Increased from 120 to 300 seconds (5 minutes)
        self.task_started_at: dict[str, float] = {}
        self.timed_out: set[str] = set()
        # Pause/queue state
        self.paused: bool = False
        self._pending_queue: list[str] = []
        # Batching window for large folders
        self.batch_window_size: int = 20
        self._batch_remaining: list[str] = []
        # Streaming directory iterator for large recursive scans
        self._dir_iter = None
        # Track failed paths for post-batch retry and limit retry rounds
        self._failed_paths: list[str] = []
        self._retry_round: int = 0
        # Track validation retry counts per image (to prevent infinite loops)
        self._validation_retries: dict[str, int] = {}
        self._max_validation_retries: int = 2  # Max 2 auto-retries per image
        self._current_splash = None  # Reference to open splash dialog
        # Vehicle knowledge base path (try local copy first, then fallback to user's JSON file)
        local_kb_path = os.path.join(_app_base_dir(), 'supercar_cheat_sheet.json')
        fallback_kb_path = r'c:\Users\tonyd\OneDrive\Desktop\json\supercar_cheat_sheet.json'
        if os.path.exists(local_kb_path):
            self.vehicle_kb_path = local_kb_path
        elif os.path.exists(fallback_kb_path):
            self.vehicle_kb_path = fallback_kb_path
        else:
            self.vehicle_kb_path = None
        # Pre-load KB to cache it
        if self.vehicle_kb_path:
            load_vehicle_knowledge_base(self.vehicle_kb_path)
        # Race metadata tracking
        self.race_data: dict = {}
        self.race_sessions: dict = {}
        self.race_track: str = ''
        self.race_event_name: str = ''
        self.timing_pdf_path: str = ''
        self.race_tagging_enabled: bool = False
        self.race_number_only_mode: bool = False
        self.selected_session: str | None = None  # User-selected session (overrides timestamp)
        self._batch_results: list[dict] = []  # Store all batch results for splash screen
        # Remember last directory browsed for dialogs
        try:
            self._last_dir = os.getcwd()
        except Exception:
            self._last_dir = str(Path.home())

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Controls - Top Row
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(16)  # Add consistent spacing between elements
        root.addLayout(controls)
        
        # Consistent label styling
        label_style = 'QLabel { color: #e2e8f0; font-size: 12px; }'
        
        # Combo box styling
        combo_style = '''
            QComboBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #64748b;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #94a3b8;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #e2e8f0;
                selection-background-color: #2563eb;
                border: 1px solid #475569;
            }
        '''
        
        # Model Selection Group
        model_label = QtWidgets.QLabel('Model:')
        model_label.setStyleSheet(label_style)
        controls.addWidget(model_label)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setStyleSheet(combo_style)
        controls.addWidget(self.model_combo)
        # Make the model dropdown a bit wider (add ~5 chars)
        try:
            self.model_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.model_combo.setMinimumContentsLength((self.model_combo.minimumContentsLength() or 20) + 5)
        except Exception:
            pass
        
        controls.addSpacing(16)  # Gap before next group
        
        # Metadata & Processing Group
        # Refresh Models button - moved here for better organization
        # Define utility button style (used by multiple buttons)
        utility_button_style = '''
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #475569;
                border-color: #64748b;
            }
            QPushButton:pressed {
                background-color: #1e293b;
            }
        '''
        self.utility_button_style = utility_button_style  # Store for later use
        self.disabled_button_style = '''
            QPushButton {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
                font-size: 12px;
            }
        '''
        refresh_btn = QtWidgets.QPushButton('Refresh Models')
        refresh_btn.setStyleSheet(utility_button_style)
        refresh_btn.setToolTip('Re-scan local models (Ollama)')
        refresh_btn.clicked.connect(self.refresh_models)
        controls.addWidget(refresh_btn)
        
        controls.addSpacing(8)  # Small gap before metadata
        
        metadata_label = QtWidgets.QLabel('Existing Metadata:')
        metadata_label.setStyleSheet(label_style)
        controls.addWidget(metadata_label)
        self.overwrite_combo = QtWidgets.QComboBox()
        self.overwrite_combo.setStyleSheet(combo_style)
        self.overwrite_combo.addItems(['skip', 'overwrite', 'ask'])
        self.overwrite_combo.setCurrentText('skip')
        controls.addWidget(self.overwrite_combo)
        
        # Checkboxes Group with improved styling (for internal use, not displayed)
        checkbox_style = '''
            QCheckBox {
                color: #e2e8f0;
                spacing: 6px;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #475569;
                border-radius: 3px;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #2563eb;
                border-color: #3b82f6;
            }
            QCheckBox::indicator:hover {
                border-color: #64748b;
            }
        '''
        
        # Create checkboxes but don't add to UI - they'll be in the menu
        self.embed_checkbox = QtWidgets.QCheckBox('Embed in JPG')
        self.embed_checkbox.setChecked(True)
        self.embed_checkbox.setStyleSheet(checkbox_style)
        
        self.recursive_checkbox = QtWidgets.QCheckBox('Recursive Scan')
        self.recursive_checkbox.setChecked(True)
        self.recursive_checkbox.setStyleSheet(checkbox_style)
        
        # Race Metadata Tagging (Premium Feature) - Merged checkbox and button into one
        controls.addSpacing(8)
        
        self.load_pdf_btn = QtWidgets.QPushButton('Event-PDF-Import')
        self.load_pdf_btn.setStyleSheet(self.utility_button_style)  # utility_button_style stored above
        self.load_pdf_btn.setToolTip('Click to load timing PDF. Once loaded, race metadata tagging will be active.')
        self.load_pdf_btn.clicked.connect(self.on_load_timing_pdf)
        controls.addWidget(self.load_pdf_btn)
        
        self.race_number_only_checkbox = QtWidgets.QCheckBox('Tag by Race Number/Session Only')
        self.race_number_only_checkbox.setChecked(False)
        self.race_number_only_checkbox.setEnabled(False)
        self.race_number_only_checkbox.setToolTip('Skip full AI identification, use PDF data only')
        self.race_number_only_checkbox.setStyleSheet(checkbox_style)
        self.race_number_only_checkbox.toggled.connect(self.on_race_number_only_toggled)
        controls.addWidget(self.race_number_only_checkbox)
        
        # PDF Status Card - will be shown when PDF is loaded
        self.pdf_status_widget = QtWidgets.QWidget()
        self.pdf_status_widget.setVisible(False)
        # Set size policy to prevent vertical expansion - should only take about 1/3 of available space
        self.pdf_status_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        self.pdf_status_widget.setMaximumHeight(60)  # Limit height to keep it compact
        pdf_status_layout = QtWidgets.QHBoxLayout(self.pdf_status_widget)
        pdf_status_layout.setContentsMargins(8, 6, 8, 6)
        pdf_status_layout.setSpacing(8)
        
        # Remove/clear button - replaces the file icon, positioned on the left
        self.pdf_remove_btn = QtWidgets.QPushButton('✕')
        self.pdf_remove_btn.setStyleSheet('''
            QPushButton {
                background-color: transparent;
                color: #ef4444;
                border: 1px solid #ef4444;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: bold;
                min-width: 24px;
                max-width: 24px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #dc2626;
                border-color: #dc2626;
            }
        ''')
        self.pdf_remove_btn.setToolTip('Remove PDF and return to defaults')
        self.pdf_remove_btn.clicked.connect(self.on_remove_pdf)
        pdf_status_layout.addWidget(self.pdf_remove_btn)
        
        # PDF info container
        pdf_info_layout = QtWidgets.QVBoxLayout()
        pdf_info_layout.setSpacing(2)
        
        self.pdf_filename_label = QtWidgets.QLabel('')
        self.pdf_filename_label.setStyleSheet('QLabel { color: #10b981; font-weight: bold; font-size: 11px; }')
        self.pdf_filename_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        pdf_info_layout.addWidget(self.pdf_filename_label)
        
        self.pdf_details_label = QtWidgets.QLabel('')
        self.pdf_details_label.setStyleSheet('QLabel { color: #93c5fd; font-size: 9px; }')
        self.pdf_details_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        self.pdf_details_label.setWordWrap(True)  # Allow text to wrap if needed
        pdf_info_layout.addWidget(self.pdf_details_label)
        
        pdf_status_layout.addLayout(pdf_info_layout)
        pdf_status_layout.addStretch()
        
        # Style the widget container
        self.pdf_status_widget.setStyleSheet('''
            QWidget {
                background-color: #064e3b;
                border: 2px solid #10b981;
                border-radius: 6px;
                padding: 4px;
            }
        ''')
        
        # Note: pdf_status_widget will be added to pdf_session_container later (moved to bottom-right panel)
        
        # Session Selection UI (shown when PDF is loaded)
        session_widget = QtWidgets.QWidget()
        session_widget.setVisible(False)
        session_layout = QtWidgets.QVBoxLayout(session_widget)
        session_layout.setContentsMargins(0, 4, 0, 4)
        session_layout.setSpacing(4)
        
        # Session dropdown
        session_label = QtWidgets.QLabel('Active Session:')
        session_label.setStyleSheet('QLabel { color: #93c5fd; font-size: 10px; }')
        session_layout.addWidget(session_label)
        
        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.setStyleSheet('''
            QComboBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox:hover {
                border-color: #64748b;
            }
            QComboBox::drop-down {
                border: none;
            }
        ''')
        self.session_combo.currentTextChanged.connect(self.on_session_selection_changed)
        session_layout.addWidget(self.session_combo)
        
        # Driver list
        driver_label = QtWidgets.QLabel('Drivers/Entrants:')
        driver_label.setStyleSheet('QLabel { color: #93c5fd; font-size: 10px; }')
        session_layout.addWidget(driver_label)
        
        self.driver_list = QtWidgets.QListWidget()
        self.driver_list.setMaximumHeight(120)
        self.driver_list.setStyleSheet('''
            QListWidget {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 4px;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 2px;
            }
        ''')
        session_layout.addWidget(self.driver_list)
        
        self.session_widget = session_widget
        # Note: session_widget will be added to pdf_session_container later (moved to bottom-right panel)
        
        # Keep the old label for backward compatibility but hide it
        self.pdf_status_label = QtWidgets.QLabel('No PDF loaded')
        self.pdf_status_label.setStyleSheet('QLabel { color: #93c5fd; font-size: 10px; }')
        self.pdf_status_label.setVisible(False)
        controls.addWidget(self.pdf_status_label)
        
        # Motorshow Mode checkbox (auto-detection) - created but not added to UI, will be in menu
        self.motorshow_checkbox = QtWidgets.QCheckBox('Motorshow Mode (auto)')
        self.motorshow_checkbox.setChecked(False)
        self.motorshow_checkbox.setToolTip('Auto-detect multi-vehicle/exhibition scenes and produce scene-first results')
        self.motorshow_checkbox.setStyleSheet(checkbox_style)
        
        # Enhanced Reject Button
        self.reject_btn = QtWidgets.QPushButton('Reject Result')
        self.reject_btn.setStyleSheet('QPushButton { background-color: #7a1f1f; color: #fff; border-radius: 4px; padding: 6px 12px; }')
        self.reject_btn.setToolTip('Reject current result and provide feedback')
        self.reject_btn.clicked.connect(self.on_reject_clicked)
        self.reject_btn.setEnabled(False)  # Initially disabled until we have results
        # Hide the global Reject button in favor of per-result Reject next to Details
        try:
            self.reject_btn.setVisible(False)
        except Exception:
            pass
        controls.addWidget(self.reject_btn)
        
        controls.addStretch(1)  # Push everything to the left

        # Actions row (put buttons on a new line to avoid truncation)
        actions = QtWidgets.QHBoxLayout()
        try:
            actions.setContentsMargins(8, 8, 8, 8)
            actions.setSpacing(12)
        except Exception:
            pass
        root.addLayout(actions)
        # Main action buttons with improved styling
        primary_button_style = '''
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        '''
        
        btn_img = QtWidgets.QPushButton('Process Image')
        btn_img.setStyleSheet(primary_button_style)
        btn_img.setToolTip('Pick a single image file to tag')
        btn_img.clicked.connect(self.select_and_process_image)
        actions.addWidget(btn_img)
        
        btn_batch = QtWidgets.QPushButton('Process Folders')
        btn_batch.setStyleSheet(primary_button_style)
        btn_batch.setToolTip('Pick one or more folders to process (respects Recursive Scan)')
        btn_batch.clicked.connect(self.select_and_process_folders)
        actions.addWidget(btn_batch)
        
        actions.addSpacing(12)  # Space between primary and control buttons

        # Emergency Stop and Pause/Resume controls with improved styling
        stop_button_style = '''
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                border: 1px solid #ef4444;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                border-color: #f87171;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        '''
        
        self.btn_stop = QtWidgets.QPushButton('Emergency Stop')
        self.btn_stop.setStyleSheet(stop_button_style)
        self.btn_stop.setToolTip('Immediately stop scheduling and cancel in-flight previews')
        self.btn_stop.clicked.connect(self.on_emergency_stop_clicked)
        actions.addWidget(self.btn_stop)

        self.btn_pause = QtWidgets.QPushButton('Pause')
        self.btn_pause.setStyleSheet(primary_button_style)
        self.btn_pause.setToolTip('Pause/resume scheduling new items (in-flight continue)')
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        actions.addWidget(self.btn_pause)
        
        # Max Concurrent control next to pause button
        actions.addSpacing(8)
        mc_label = QtWidgets.QLabel('Max Concurrent:')
        mc_label.setStyleSheet('QLabel { color: #e2e8f0; font-size: 12px; }')
        mc_label.setToolTip('Number of simultaneous image inferences')
        actions.addWidget(mc_label)
        self.workers_combo = QtWidgets.QComboBox()
        self.workers_combo.setStyleSheet('''
            QComboBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #475569;
            }
            QComboBox::drop-down {
                border: none;
            }
        ''')
        self.workers_combo.addItems(['1','2','3','4'])
        try:
            self.workers_combo.setFixedWidth(70)
        except Exception:
            pass
        self.workers_combo.currentTextChanged.connect(self.on_workers_combo_changed)
        actions.addWidget(self.workers_combo)

        # Add stretch to push logo to the right
        actions.addStretch(1)
        
        # Utility buttons group with consistent styling
        utility_button_style_actions = '''
            QPushButton {
                background-color: #1e40af;
                color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1e3a8a;
            }
        '''
        
        # Create dropdown menu for About, Requirements, Help, and Export KB
        menu_btn = QtWidgets.QPushButton('Menu ▼')
        menu_btn.setStyleSheet(utility_button_style_actions)
        menu_btn.setToolTip('Additional options and information')
        menu = QtWidgets.QMenu(menu_btn)
        # Style the menu with hover effects similar to combo box dropdown
        menu.setStyleSheet('''
            QMenu {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 8px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QMenu::item:hover {
                background-color: #3b82f6;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #475569;
                margin: 4px 8px;
            }
        ''')
        
        about_action = menu.addAction('About')
        about_action.triggered.connect(self.show_about)
        
        req_action = menu.addAction('Requirements')
        req_action.triggered.connect(self.show_requirements)
        
        help_action = menu.addAction('Help')
        help_action.triggered.connect(self.show_help)
        
        menu.addSeparator()
        
        # Processing Options Section
        embed_action = menu.addAction('Embed in JPG')
        embed_action.setCheckable(True)
        embed_action.setChecked(True)
        embed_action.triggered.connect(lambda checked: self.embed_checkbox.setChecked(checked))
        self.embed_checkbox.toggled.connect(lambda checked: embed_action.setChecked(checked))
        
        recursive_action = menu.addAction('Recursive Scan')
        recursive_action.setCheckable(True)
        recursive_action.setChecked(True)
        recursive_action.triggered.connect(lambda checked: self.recursive_checkbox.setChecked(checked))
        self.recursive_checkbox.toggled.connect(lambda checked: recursive_action.setChecked(checked))
        
        motorshow_action = menu.addAction('Motorshow Mode (auto)')
        motorshow_action.setCheckable(True)
        motorshow_action.setChecked(False)
        motorshow_action.setToolTip('Auto-detect multi-vehicle/exhibition scenes and produce scene-first results')
        motorshow_action.triggered.connect(lambda checked: self.motorshow_checkbox.setChecked(checked))
        self.motorshow_checkbox.toggled.connect(lambda checked: motorshow_action.setChecked(checked))
        
        menu.addSeparator()
        
        export_kb_action = menu.addAction('Export KB')
        export_kb_action.setToolTip('Export user knowledge base for submission')
        export_kb_action.triggered.connect(self.on_export_kb_clicked)
        
        menu_btn.setMenu(menu)
        actions.addWidget(menu_btn)
        
        # KB Manager button (kept separate as it's frequently used)
        kb_manager_btn = QtWidgets.QPushButton('KB Manager')
        kb_manager_btn.setStyleSheet(utility_button_style_actions)
        kb_manager_btn.setToolTip('View and manage user knowledge base entries')
        kb_manager_btn.clicked.connect(self.show_kb_manager)
        actions.addWidget(kb_manager_btn)
        
        actions.addSpacing(12)
        
        # Logo for hidden menu (right-click for cloud settings) - as large as possible
        try:
            self.header_logo = QtWidgets.QLabel()
            _pix_hdr = QtGui.QPixmap(self._app_logo_path)
            if not _pix_hdr.isNull():
                # Make it as large as possible while fitting in the button row (200x200 for maximum visibility)
                self.header_logo.setPixmap(_pix_hdr.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.header_logo.setToolTip('Right-click for advanced settings')
                self.header_logo.setStyleSheet('QLabel { background-color: transparent; padding: 0px; }')
                # Hidden access: right-click logo to open cloud settings (password gate)
                self.header_logo.setContextMenuPolicy(Qt.CustomContextMenu)
                self.header_logo.customContextMenuRequested.connect(self._on_logo_context)
                actions.addWidget(self.header_logo)
                actions.addSpacing(12)
        except Exception:
            pass

        # Now Processing panel (max 4)
        proc_group = QtWidgets.QGroupBox('Now Processing')
        proc_layout = QtWidgets.QGridLayout(proc_group)
        self.processing_slots: list[QtWidgets.QLabel] = []
        for i in range(4):
            lbl = QtWidgets.QLabel()
            lbl.setMinimumSize(220, 165)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet('background:#0f172a; color:#93c5fd; border:1px solid #1e293b; border-radius:4px;')
            lbl.setText('Idle')
            lbl.setProperty('busy', False)
            proc_layout.addWidget(lbl, i // 2, i % 2)
            self.processing_slots.append(lbl)
        root.addWidget(proc_group)
        # Apply initial visibility based on concurrency setting
        self._update_processing_slots_visibility()
        
        # PDF Summary and Session container - will be in bottom splitter (initially hidden)
        self.pdf_session_container = QtWidgets.QFrame()
        self.pdf_session_container.setVisible(False)
        self.pdf_session_container.setStyleSheet('''
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px;
            }
        ''')
        pdf_session_layout = QtWidgets.QVBoxLayout(self.pdf_session_container)
        pdf_session_layout.setContentsMargins(8, 8, 8, 8)
        pdf_session_layout.setSpacing(8)
        
        # Move PDF status widget into this container
        pdf_session_layout.addWidget(self.pdf_status_widget)
        
        # Move session widget into this container
        pdf_session_layout.addWidget(self.session_widget)

        # Bottom viewport: Results list (2/3) + PDF Summary/Session (1/3) split horizontally
        bottom_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        
        # Results list - left side (2/3)
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setSortingEnabled(False)  # Ensure no automatic sorting
        bottom_splitter.addWidget(self.results_list)
        
        # PDF Summary/Session container - right side (1/3)
        bottom_splitter.addWidget(self.pdf_session_container)
        
        # Set splitter sizes: 2/3 for results, 1/3 for PDF (ratio 2:1)
        bottom_splitter.setStretchFactor(0, 2)  # Results list
        bottom_splitter.setStretchFactor(1, 1)  # PDF container
        bottom_splitter.setSizes([666, 334])  # Initial sizes (2:1 ratio)
        
        # Store splitter reference for later use
        self.bottom_splitter = bottom_splitter
        
        root.addWidget(bottom_splitter, 1)

        # Allocate vertical space: controls (0), actions (1), processing (2), bottom_splitter (3)
        try:
            root.setStretch(0, 0)  # controls row
            root.setStretch(1, 0)  # actions row
            root.setStretch(2, 1)  # processing group gets 50%
            root.setStretch(3, 1)  # bottom splitter (results + PDF) gets 50%
        except Exception:
            pass

        # Status
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        # Bottom-left persistent eye icon (plays only when inference is active)
        try:
            self.eye_label = QtWidgets.QLabel()
            self.eye_movie = QtGui.QMovie(_resource_path('eye.gif'))
            try:
                self.eye_movie.setScaledSize(QtCore.QSize(24, 24))
            except Exception:
                pass
            self.eye_label.setMovie(self.eye_movie)
            # Start and immediately pause so first frame is shown statically
            self.eye_movie.start()
            try:
                self.eye_movie.setPaused(True)
            except Exception:
                pass
            self.status.addWidget(self.eye_label)
        except Exception:
            self.eye_movie = None
            pass
        # Build a compact, styled right-side status cluster
        self.connection_label = QtWidgets.QLabel('🔴 Disconnected')
        self.remaining_label = QtWidgets.QLabel('Remaining: --')
        self.avg_label = QtWidgets.QLabel('Avg last 3: --')
        self.inference_mode_label = QtWidgets.QLabel('Mode: Car Analysis')
        self.inference_mode_label.setStyleSheet('QLabel { background-color: #0c1a33; border: 1px solid #2563eb; border-radius: 10px; padding: 2px 10px; color: #87ceeb; font-weight: bold; }')
        self.progress = QtWidgets.QProgressBar()
        # Pulsing eye activity icon (fixed 16x16, right side)
        self.activity_label = QtWidgets.QLabel()
        try:
            self.activity_movie = QtGui.QMovie(_resource_path('eye.gif'))
            try:
                self.activity_movie.setScaledSize(QtCore.QSize(16, 16))
            except Exception:
                pass
            self.activity_label.setMovie(self.activity_movie)
            self.activity_label.setFixedSize(16, 16)
            self.activity_label.setVisible(False)
        except Exception:
            self.activity_movie = None
            self.activity_label.setText('')
        try:
            self.progress.setFixedWidth(240)
        except Exception:
            pass
        self.progress.setVisible(False)
        status_container = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_container)
        try:
            status_layout.setContentsMargins(0, 0, 0, 0)
            status_layout.setSpacing(8)
        except Exception:
            pass
        # Badge styles for a sleeker look
        try:
            self.connection_label.setStyleSheet('QLabel { background-color: #0c1a33; border: 1px solid #2563eb; border-radius: 10px; padding: 2px 10px; color: #cfe3ff; }')
            self.remaining_label.setStyleSheet('QLabel { background-color: #0c1a33; border: 1px solid #334155; border-radius: 10px; padding: 2px 10px; color: #cfe3ff; }')
            self.avg_label.setStyleSheet('QLabel { background-color: #0c1a33; border: 1px solid #334155; border-radius: 10px; padding: 2px 10px; color: #cfe3ff; }')
        except Exception:
            pass
        status_layout.addWidget(self.connection_label)
        status_layout.addWidget(self.remaining_label)
        status_layout.addWidget(self.avg_label)
        status_layout.addWidget(self.inference_mode_label)
        status_layout.addWidget(self.progress)
        status_layout.addStretch(1)
        try:
            self.status.addPermanentWidget(status_container, 1)
        except Exception:
            # Fallback: add individually if container fails
            self.status.addPermanentWidget(self.connection_label)
            self.status.addPermanentWidget(self.avg_label)
            self.status.addPermanentWidget(self.inference_mode_label)
        self.status.addPermanentWidget(self.progress)
        self.recent_durations: list[float] = []
        # Accumulate feedback for potential future weighting/analytics
        self._feedback_log: list[dict] = []

        # Helper to toggle the eye animation
        def _set_activity(active: bool) -> None:
            try:
                if getattr(self, 'eye_movie', None):
                    if active:
                        # play
                        try:
                            self.eye_movie.setPaused(False)
                        except Exception:
                            self.eye_movie.start()
                    else:
                        # pause on first frame
                        try:
                            self.eye_movie.setPaused(True)
                            self.eye_movie.jumpToFrame(0)
                        except Exception:
                            self.eye_movie.stop()
                # Keep the right-side activity label hidden
                try:
                    if hasattr(self, 'activity_label'):
                        self.activity_label.setVisible(False)
                except Exception:
                    pass
            except Exception:
                pass
        self._set_activity = _set_activity
        
        # Simple console+file logger (write log where the EXE is run from)
        try:
            base_dir = None
            try:
                import sys as _sys
                if getattr(_sys, 'frozen', False) and getattr(_sys, 'executable', ''):
                    base_dir = Path(_sys.executable).parent
            except Exception:
                base_dir = None
            if base_dir is None:
                try:
                    base_dir = Path.cwd()
                except Exception:
                    base_dir = Path(__file__).parent
            self.log_file_path = (base_dir / 'log.log')
        except Exception:
            self.log_file_path = Path('log.log')
        def _console_log(msg: str):
            try:
                timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
                print(timestamped, flush=True)
                try:
                    with open(self.log_file_path, 'a', encoding='utf-8') as lf:
                        lf.write(timestamped + "\n")
                except Exception:
                    pass
            except Exception:
                pass
        self._console_log = _console_log

        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        # Defer network-bound model discovery until after the window shows
        QtCore.QTimer.singleShot(0, self.refresh_models)

        # Watchdog timer for timeouts
        self.watchdog = QtCore.QTimer(self)
        self.watchdog.setInterval(1000)
        self.watchdog.timeout.connect(self._check_timeouts)
        self.watchdog.start()
        # Lightweight scheduler safeguard: periodically try to drain if capacity exists
        try:
            self._drain_tick = QtCore.QTimer(self)
            self._drain_tick.setInterval(500)
            self._drain_tick.timeout.connect(self._drain_queue)
            self._drain_tick.start()
        except Exception:
            pass
        # Apply blue accent theme (removes any orange accents)
        try:
            self._apply_blue_theme()
        except Exception:
            pass
        # Init settings loader
        try:
            self._load_cloud_settings()
        except Exception:
            pass
        # Load persisted max concurrent preference
        try:
            s = QSettings('Pistonspy', 'TagOmatic')
            saved = int(s.value('ui/max_concurrent', 1, type=int))
            saved = min(4, max(1, saved))
            self.workers_combo.setCurrentText(str(saved))
            # Apply the loaded setting to internal variables
            self.max_concurrent = saved
            self.dynamic_limit = saved
        except Exception:
            pass
        
        # Version check moved to main() function to avoid splash screen interference

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        # Auto-cleanup unreferenced KB images on startup (silently, in background)
        try:
            cleanup_unreferenced_kb_images()
        except Exception:
            pass  # Silent failure - don't interrupt startup
        try:
            if not getattr(self, '_faded_in', False):
                effect = QtWidgets.QGraphicsOpacityEffect(self)
                effect.setOpacity(0.0)
                self.setGraphicsEffect(effect)
                anim = QtCore.QPropertyAnimation(effect, b'opacity', self)
                anim.setDuration(400)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
                anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
                anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
                self._faded_in = True
        except Exception:
            pass
        # Version check moved to main() function to avoid splash screen interference
        return super().showEvent(event)

    @Slot()
    def show_about(self):
        try:
            dlg = AboutDialog(getattr(self, '_app_logo_path', ''), self)
            dlg.exec()
        except Exception:
            QtWidgets.QMessageBox.information(self, 'About', 'Pistonspy : TagOmatic')

    @Slot()
    def show_requirements(self):
        try:
            dlg = RequirementsDialog(self)
            dlg.exec()
        except Exception:
            QtWidgets.QMessageBox.information(self, 'Requirements', 'Install Ollama and pull models: qwen2.5vl:7b, gemma3:12b')
    
    @Slot()
    def show_help(self):
        try:
            dlg = HelpDialog(self)
            dlg.exec()
        except Exception:
            QtWidgets.QMessageBox.information(self, 'Help', 'Comprehensive usage instructions are available in the Help dialog.')
    
    def on_export_kb_clicked(self):
        """Export user knowledge base as encrypted ZIP for submission."""
        try:
            user_kb_path = os.path.join(_app_base_dir(), 'user_knowledge_base.json')
            if not os.path.exists(user_kb_path):
                QtWidgets.QMessageBox.information(self, 'No KB', 'No user knowledge base found to export.')
                return
            
            # Get output path
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Export Knowledge Base',
                os.path.join(_app_base_dir(), 'kb_export.encrypted'),
                'Encrypted Files (*.encrypted);;All Files (*.*)'
            )
            
            if not output_path:
                return
            
            # Export
            if export_knowledge_base_for_submission(output_path):
                QtWidgets.QMessageBox.information(
                    self,
                    'Export Success',
                    f'Knowledge base exported successfully to:\n{output_path}\n\nYou can now submit this file to the developer.'
                )
            else:
                QtWidgets.QMessageBox.warning(self, 'Export Failed', 'Failed to export knowledge base.')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Error exporting KB: {e}')
            _log(f"Error in on_export_kb_clicked: {e}")
            import traceback
            _log(traceback.format_exc())
    
    def show_kb_manager(self):
        """Open the Knowledge Base Manager dialog."""
        try:
            dlg = KnowledgeBaseManagerDialog(self)
            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Error opening KB Manager: {e}')
            _log(f"Error in show_kb_manager: {e}")
            import traceback
            _log(traceback.format_exc())
    
    def on_race_tagging_toggled(self, checked: bool):
        """Handle race tagging checkbox toggle - no longer used, kept for compatibility."""
        # This method is kept for compatibility but checkbox is removed
        # Race tagging is now enabled automatically when PDF is loaded
        pass
    
    def on_race_number_only_toggled(self, checked: bool):
        """Handle race number only mode toggle."""
        self.race_number_only_mode = checked
    
    def get_filtered_race_data(self, session: str | None) -> dict:
        """Filter race data by session.
        
        Args:
            session: Session name to filter by, or None/"All Sessions" for all data
        
        Returns:
            Filtered race data dict
        """
        if not session or session == 'All Sessions':
            return self.race_data
        
        # Filter to only entries matching the session
        filtered = {}
        for race_number, data in self.race_data.items():
            if data.get('session') == session:
                filtered[race_number] = data
        
        return filtered
    
    def get_active_session(self, image_path: str) -> str | None:
        """Determine active session for image processing.
        
        Priority:
        1. Manual selection (if user selected from dropdown)
        2. EXIF timestamp auto-detection
        3. None (use all sessions)
        
        Args:
            image_path: Path to image file
        
        Returns:
            Session name or None
        """
        # Override: Manual selection
        if self.selected_session and self.selected_session != 'All Sessions':
            return self.selected_session
        
        # Primary: EXIF timestamp matching
        matched_session = match_image_to_session(image_path, self.race_sessions)
        if matched_session:
            return matched_session
        
        # Fallback: None (use all sessions)
        return None
    
    def update_driver_list(self):
        """Update driver list widget with entrants for selected session."""
        self.driver_list.clear()
        
        if not self.race_data:
            return
        
        # Get selected session
        selected = self.session_combo.currentText()
        if selected == 'All Sessions':
            # Show all drivers
            filtered_data = self.race_data
        else:
            filtered_data = self.get_filtered_race_data(selected)
        
        # Build driver list
        drivers = []
        for race_number in sorted(filtered_data.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            data = filtered_data[race_number]
            driver = data.get('driver', 'N/A')
            team = data.get('team', 'N/A')
            car = data.get('car', 'N/A')
            drivers.append(f"#{race_number} | {driver} | {team} | {car}")
        
        # Add to list widget
        for driver_info in drivers:
            self.driver_list.addItem(driver_info)
    
    def on_session_selection_changed(self, text: str):
        """Handle session dropdown selection change."""
        if text == 'All Sessions':
            self.selected_session = None
        else:
            self.selected_session = text
        self.update_driver_list()
    
    def on_load_timing_pdf(self):
        """Load and parse timing PDF for race metadata."""
        try:
            pdf_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                'Load Timing PDF',
                self._last_dir if hasattr(self, '_last_dir') else '',
                'PDF Files (*.pdf);;All Files (*.*)'
            )
            
            if not pdf_path:
                return
            
            self.timing_pdf_path = pdf_path
            self._last_dir = os.path.dirname(pdf_path)
            
            # Import and parse PDF
            from race_metadata_parser import parse_timing_pdf
            self.race_data, self.race_sessions, self.race_track, self.race_event_name = parse_timing_pdf(pdf_path)
            
            # Update UI with cool visual display
            entry_count = len(self.race_data)
            session_count = len(self.race_sessions)
            pdf_filename = os.path.basename(pdf_path)
            
            # Update the visual PDF status widget
            self.pdf_filename_label.setText(f'✓ {pdf_filename}')
            
            # Build details text
            details_parts = []
            details_parts.append(f'{entry_count} entries')
            details_parts.append(f'{session_count} sessions')
            if self.race_track:
                details_parts.append(f'🏁 {self.race_track}')
            if self.race_event_name:
                details_parts.append(f'🎯 {self.race_event_name}')
            
            self.pdf_details_label.setText(' • '.join(details_parts))
            
            # Disable button and show PDF container in bottom splitter
            self.load_pdf_btn.setEnabled(False)
            self.load_pdf_btn.setStyleSheet(self.disabled_button_style)
            self.pdf_status_widget.setVisible(True)
            self.session_widget.setVisible(True)
            self.pdf_session_container.setVisible(True)  # Show the container in bottom splitter
            # Restore splitter sizes to 2:1 ratio when PDF is shown
            self.bottom_splitter.setSizes([666, 334])
            self.pdf_status_label.setVisible(False)
            
            # Keep old label updated for backward compatibility
            status_text = f'PDF loaded: {entry_count} entries, {session_count} sessions'
            if self.race_track:
                status_text += f', Track: {self.race_track}'
            if self.race_event_name:
                status_text += f', Event: {self.race_event_name}'
            self.pdf_status_label.setText(status_text)
            
            # Populate session dropdown
            self.session_combo.clear()
            self.session_combo.addItem('All Sessions')
            for session_name in sorted(self.race_sessions.keys()):
                self.session_combo.addItem(session_name)
            
            # Session widget visibility already set above
            self.selected_session = None  # Reset to auto-detect mode
            self.update_driver_list()
            
            # Enable race tagging automatically when PDF is loaded
            self.race_tagging_enabled = True
            self.race_number_only_checkbox.setEnabled(entry_count > 0)
            
            _log(f"[PDF-LOAD] Loaded {entry_count} race entries and {session_count} sessions from {os.path.basename(pdf_path)}")
            if self.race_track:
                _log(f"[PDF-LOAD] Track identified: {self.race_track}")
            if self.race_event_name:
                _log(f"[PDF-LOAD] Event identified: {self.race_event_name}")
            
            # Show a dialog with the race data table instead of simple message box
            self._show_pdf_data_table(pdf_path, entry_count, session_count)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load PDF: {e}')
            _log(f"Error loading PDF: {e}")
            import traceback
            _log(traceback.format_exc())
            # Reset UI on error
            self.load_pdf_btn.setEnabled(True)
            self.load_pdf_btn.setStyleSheet(self.utility_button_style)
            self.pdf_session_container.setVisible(False)
            self.pdf_status_widget.setVisible(False)
            self.session_widget.setVisible(False)
            # Make results list take full width when PDF is hidden
            self.bottom_splitter.setSizes([1000, 0])
    
    def on_remove_pdf(self):
        """Remove loaded PDF and return to defaults."""
        # Clear PDF data
        self.timing_pdf_path = None
        self.race_data = {}
        self.race_sessions = {}
        self.race_track = ''
        self.race_event_name = ''
        self.selected_session = None
        
        # Reset UI: hide PDF container, re-enable button
        self.pdf_session_container.setVisible(False)
        self.pdf_status_widget.setVisible(False)
        self.session_widget.setVisible(False)
        # Make results list take full width when PDF is hidden
        self.bottom_splitter.setSizes([1000, 0])
        self.load_pdf_btn.setEnabled(True)
        self.load_pdf_btn.setText('Event-PDF-Import')
        self.load_pdf_btn.setToolTip('Click to load timing PDF. Once loaded, race metadata tagging will be active.')
        self.load_pdf_btn.setStyleSheet(self.utility_button_style)  # Reset to default style
        
        # Clear PDF status labels
        self.pdf_filename_label.setText('')
        self.pdf_details_label.setText('')
        self.pdf_status_label.setText('No PDF loaded')
        self.pdf_status_label.setVisible(False)
        
        # Hide session widget
        self.session_widget.setVisible(False)
        self.session_combo.clear()
        self.driver_list.clear()
        
        # Disable race tagging
        self.race_tagging_enabled = False
        self.race_number_only_checkbox.setEnabled(False)
        self.race_number_only_checkbox.setChecked(False)
        
        _log("[PDF-LOAD] PDF removed, returned to defaults")
    
    def _show_pdf_data_table(self, pdf_path: str, entry_count: int, session_count: int):
        """Display PDF race data in a nice table dialog."""
        try:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(f'PDF Data: {os.path.basename(pdf_path)}')
            dlg.resize(1000, 700)
            
            layout = QtWidgets.QVBoxLayout(dlg)
            
            # Header with summary info
            header_text = f'<h3 style="color:#93c5fd; margin:0;">PDF Loaded Successfully</h3>'
            header_text += f'<p style="color:#cbd5e1; margin:5px 0;">'
            header_text += f'<b>{entry_count}</b> race entries • <b>{session_count}</b> sessions'
            if self.race_track:
                header_text += f' • Track: <b>{self.race_track}</b>'
            if self.race_event_name:
                header_text += f' • Event: <b>{self.race_event_name}</b>'
            header_text += '</p>'
            
            header_label = QtWidgets.QLabel(header_text)
            header_label.setStyleSheet('QLabel { padding: 10px; background-color: #1e293b; border-radius: 4px; }')
            layout.addWidget(header_label)
            
            # Create tab widget for different views
            tabs = QtWidgets.QTabWidget()
            tabs.setStyleSheet('''
                QTabWidget::pane {
                    border: 1px solid #475569;
                    background-color: #0f172a;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #1e293b;
                    color: #cbd5e1;
                    padding: 8px 16px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QTabBar::tab:hover {
                    background-color: #334155;
                }
            ''')
            
            # Race Entries Table Tab
            entries_tab = QtWidgets.QWidget()
            entries_layout = QtWidgets.QVBoxLayout(entries_tab)
            
            entries_table = QtWidgets.QTableWidget()
            entries_table.setColumnCount(5)
            entries_table.setHorizontalHeaderLabels(['Race #', 'Car', 'Team', 'Driver', 'Session'])
            entries_table.setStyleSheet('''
                QTableWidget {
                    background-color: #0f172a;
                    color: #e2e8f0;
                    border: 1px solid #334155;
                    gridline-color: #334155;
                    font-size: 11px;
                }
                QTableWidget::item {
                    padding: 4px;
                    border: none;
                }
                QTableWidget::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background-color: #1e293b;
                    color: #93c5fd;
                    padding: 6px;
                    border: none;
                    font-weight: bold;
                    font-size: 11px;
                }
                QTableWidget::item:hover {
                    background-color: #1e40af;
                }
            ''')
            
            # Populate race entries table
            entries_table.setRowCount(entry_count)
            row = 0
            for race_number, data in sorted(self.race_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                entries_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(race_number)))
                entries_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(data.get('car', ''))))
                entries_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(data.get('team', ''))))
                entries_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(data.get('driver', ''))))
                entries_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(data.get('session', ''))))
                row += 1
            
            entries_table.resizeColumnsToContents()
            entries_table.setAlternatingRowColors(True)
            entries_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            entries_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            
            entries_layout.addWidget(entries_table)
            tabs.addTab(entries_tab, f'Race Entries ({entry_count})')
            
            # Sessions Table Tab
            if session_count > 0:
                sessions_tab = QtWidgets.QWidget()
                sessions_layout = QtWidgets.QVBoxLayout(sessions_tab)
                
                sessions_table = QtWidgets.QTableWidget()
                sessions_table.setColumnCount(3)
                sessions_table.setHorizontalHeaderLabels(['Session', 'Start Time', 'End Time'])
                sessions_table.setStyleSheet(entries_table.styleSheet())
                
                # Populate sessions table
                sessions_table.setRowCount(session_count)
                row = 0
                for session_name, (start_dt, end_dt) in sorted(self.race_sessions.items()):
                    sessions_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(session_name)))
                    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S') if start_dt else 'N/A'
                    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S') if end_dt else 'N/A'
                    sessions_table.setItem(row, 1, QtWidgets.QTableWidgetItem(start_str))
                    sessions_table.setItem(row, 2, QtWidgets.QTableWidgetItem(end_str))
                    row += 1
                
                sessions_table.resizeColumnsToContents()
                sessions_table.setAlternatingRowColors(True)
                sessions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
                sessions_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
                
                sessions_layout.addWidget(sessions_table)
                tabs.addTab(sessions_tab, f'Sessions ({session_count})')
            
            layout.addWidget(tabs)
            
            # Close button
            btn_layout = QtWidgets.QHBoxLayout()
            btn_layout.addStretch()
            close_btn = QtWidgets.QPushButton('Close')
            close_btn.setStyleSheet('''
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #3b82f6;
                    border-radius: 4px;
                    padding: 8px 20px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #3b82f6;
                }
            ''')
            close_btn.clicked.connect(dlg.accept)
            btn_layout.addWidget(close_btn)
            layout.addLayout(btn_layout)
            
            dlg.exec()
            
        except Exception as e:
            _log(f"Error showing PDF data table: {e}")
            import traceback
            _log(traceback.format_exc())
            # Fallback to simple message box
            QtWidgets.QMessageBox.information(
                self,
                'PDF Loaded',
                f'Timing PDF loaded successfully.\n\n{entry_count} race entries\n{session_count} sessions'
            )

    def _on_logo_context(self, _pos):
        try:
            # Persist session unlock
            if not getattr(self, '_secret_unlocked', False):
                ok = QtWidgets.QInputDialog.getText(self, 'Verify', 'Enter password:', QtWidgets.QLineEdit.Password)
                if isinstance(ok, tuple):
                    pwd, accepted = ok
                    if not accepted:
                        return
                else:
                    return
                if str(pwd).strip() != 'Thankyou!':
                    return
                # Mark unlocked for this session
                self._secret_unlocked = True
            self._show_cloud_settings_dialog()
        except Exception:
            pass

    def _load_cloud_settings(self):
        # Prefer secrets.enc in app folder; fallback to QSettings for legacy
        use_cloud = False
        try:
            p = _secrets_path()
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    blob = f.read()
                # Windows DPAPI unwrap; on non-Windows this will be raw
                raw = _win_dpapi_unprotect(blob)
                obj = json.loads(raw.decode('utf-8', errors='replace')) if raw else {}
                if isinstance(obj, dict):
                    use_cloud = bool(obj.get('use_cloud', False))
        except Exception:
            pass
        s = QSettings('Pistonspy', 'TagOmatic')
        use_cloud = s.value('cloud/use_cloud', False, type=bool)
        preferred_provider = s.value('cloud/preferred_provider', 'auto', type=str)
        # Extended providers (no OpenAI)
        ext_ollama_host = s.value('cloud/ollama_host', 'http://localhost:11434', type=str)
        ext_ollama_model = s.value('cloud/ollama_model', '', type=str)
        oai_compat_base = s.value('cloud/oai_compat_base_url', 'http://localhost:1234/v1', type=str)
        oai_compat_api = s.value('cloud/oai_compat_api_key', '', type=str)
        oai_compat_model = s.value('cloud/oai_compat_model', '', type=str)
        cloud_router.configure(
            use_cloud=use_cloud,
            preferred_provider=preferred_provider,
            external_ollama_host=ext_ollama_host,
            external_ollama_model=ext_ollama_model,
            oai_compat_base_url=oai_compat_base,
            oai_compat_api_key=oai_compat_api,
            oai_compat_model=oai_compat_model,
        )
        # Reflect provider in the status bar
        try:
            self._update_connection_label()
        except Exception:
            pass
        # If cloud is preferred, quickly test connectivity in background
        try:
            if use_cloud and (cloud_router.remote_ollama.is_configured() or cloud_router.openai_compat.is_configured()):
                self._test_cloud_connectivity_async()
        except Exception:
            pass
    def _update_connection_label(self):
        """Show which backend will be used and make it obvious when external/cloud is active."""
        try:
            s = QSettings('Pistonspy', 'TagOmatic')
            use_cloud = s.value('cloud/use_cloud', False, type=bool)
            provider = s.value('cloud/preferred_provider', 'auto', type=str)

            if use_cloud:
                # OpenAI-compatible (e.g., LLMStudio)
                if provider == 'openai-compatible':
                    base = s.value('cloud/oai_compat_base_url', 'http://localhost:1234/v1', type=str)
                    mdl = s.value('cloud/oai_compat_model', '(pick model)', type=str)
                    self.connection_label.setText(f'🟢 Connected - OpenAI‑Compatible @ {base} · {mdl}')
                    try:
                        if hasattr(self, 'model_combo'):
                            if not hasattr(self, '_prev_model_text'):
                                self._prev_model_text = self.model_combo.currentText()
                            self.model_combo.setEditable(True)
                            self.model_combo.setCurrentText('(None - Cloud)')
                            self.model_combo.setEnabled(False)
                    except Exception:
                        pass
                    return
                # Remote Ollama host
                elif provider == 'remote-ollama':
                    host = s.value('cloud/ollama_host', 'http://localhost:11434', type=str)
                    mdl = s.value('cloud/ollama_model', '(pick model)', type=str)
                    self.connection_label.setText(f'🟢 Connected - Ollama @ {host} · {mdl}')
                    try:
                        if hasattr(self, 'model_combo'):
                            if not hasattr(self, '_prev_model_text'):
                                self._prev_model_text = self.model_combo.currentText()
                            self.model_combo.setEditable(True)
                            self.model_combo.setCurrentText('(None - Cloud)')
                            self.model_combo.setEnabled(False)
                    except Exception:
                        pass
                    return
                else: # Auto or unspecified
                    # Fallback display for auto mode
                    if cloud_router.openai_compat.is_configured():
                        self.connection_label.setText(f'🟢 Connected - OpenAI-Compatible (Auto)')
                    elif cloud_router.remote_ollama.is_configured():
                        self.connection_label.setText(f'🟢 Connected - Remote Ollama (Auto)')
                    else:
                        self.connection_label.setText(f'🟢 Cloud Active (Auto)')
                    return
        except Exception:
            pass

        # ---- LOCAL MODEL STATUS ----
        # This code now only runs if use_cloud was false.
        # Restore local model selectors when not in cloud mode
        try:
            if hasattr(self, 'model_combo'):
                self.model_combo.setEnabled(True)
                # Restore previous text if we changed it
                if hasattr(self, '_prev_model_text') and self._prev_model_text:
                    # Try to pick from list; otherwise just set text
                    try:
                        self.model_combo.setEditable(False)
                        self.model_combo.setCurrentText(self._prev_model_text)
                        self._prev_model_text = None # Clear after restoring
                    except Exception:
                        self.model_combo.setEditable(True)
                        self.model_combo.setCurrentText(self._prev_model_text or 'mistral-small')
        except Exception:
            pass
        # Sync local model from dropdown if available and update label
        try:
            if hasattr(self, 'model_combo'):
                current = self.model_combo.currentText().strip()
                if current and current != '(None - Cloud)':
                    self.model_name = current
        except Exception:
            pass
        self.connection_label.setText(f'🟢 Connected - {self.model_name}')

    def _show_motorshow_settings_dialog(self):
        s = QSettings('Pistonspy', 'TagOmatic')
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Motorshow Mode Settings')
        dlg.setModal(True)
        vbox = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        motorshow_cb = QtWidgets.QCheckBox('Enable Motorshow Mode')
        motorshow_cb.setChecked(s.value('motorshow/enabled', False, type=bool))
        form.addRow(motorshow_cb)
        vbox.addLayout(form)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        vbox.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            s.setValue('motorshow/enabled', motorshow_cb.isChecked())
            s.sync()

    def _make_test_image_b64(self) -> str:
        """Create a tiny valid PNG as base64 for connectivity tests."""
        try:
            img = Image.new('RGB', (64, 64), color=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception:
            return ''

    def _test_cloud_connectivity_async(self):
        """Send a minimal ping to the selected cloud provider; show error if invalid."""
        def _do_test():
            try:
                # Text-only minimal test to avoid image-related gating
                messages = [{'role': 'user', 'content': 'Say OK once'}]
                res = cloud_router.chat(messages=messages, images_b64=None, options_override={'num_predict': 16})
                text = extract_message_text(res)
                if isinstance(text, str) and text.strip():
                    prov = res.get('_provider', 'cloud') if isinstance(res, dict) else 'cloud'
                    self._console_log(f"[CLOUD TEST] OK via {prov}")
                    QtCore.QMetaObject.invokeMethod(self.status, 'showMessage', Qt.QueuedConnection,
                                                    QtCore.Q_ARG(str, f'Cloud test OK: {prov}'),
                                                    QtCore.Q_ARG(int, 4000))
                else:
                    snippet = ''
                    try:
                        import json as _json
                        snippet = (res if isinstance(res, str) else _json.dumps(res))
                        snippet = snippet[:300]
                    except Exception:
                        pass
                    raise RuntimeError(f'Empty response from cloud provider. Raw: {snippet}')
            except Exception as e:
                msg = f'Cloud test failed: {e}'
                self._console_log(f"[CLOUD TEST FAIL] {msg}")
                def _warn():
                    QtWidgets.QMessageBox.warning(self, 'Cloud Test Failed', msg)
                    self.status.showMessage(msg, 6000)
                QtCore.QTimer.singleShot(0, _warn)
        th = threading.Thread(target=_do_test, daemon=True)
        th.start()

    def _show_cloud_settings_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Cloud Backends (Advanced)')
        dlg.setModal(True)
        vbox = QtWidgets.QVBoxLayout(dlg)
        form_top = QtWidgets.QFormLayout()
        use_cloud_cb = QtWidgets.QCheckBox('Prefer cloud when available')
        provider_combo = QtWidgets.QComboBox()
        provider_combo.addItems(['auto', 'remote-ollama', 'openai-compatible'])
        form_top.addRow(use_cloud_cb)
        form_top.addRow('Preferred provider:', provider_combo)
        vbox.addLayout(form_top)

        # Provider-specific stacked forms
        stack = QtWidgets.QStackedWidget()

        # External Ollama pane (index 0)
        pane_ollama = QtWidgets.QWidget()
        foa = QtWidgets.QFormLayout(pane_ollama)
        ollama_host_le = QtWidgets.QLineEdit()
        ollama_host_le.setPlaceholderText('http://host-or-ip:11434')
        ollama_model_le = QtWidgets.QComboBox()
        btn_ollama_refresh = QtWidgets.QPushButton('Refresh Models')
        btn_ollama_test = QtWidgets.QPushButton('Test Ollama')
        foa.addRow('Host:', ollama_host_le)
        foa.addRow('Model:', ollama_model_le)
        foa.addRow(btn_ollama_refresh, btn_ollama_test)
        stack.addWidget(pane_ollama)

        # OpenAI-compatible (LM Studio, etc.) pane (index 1)
        pane_oai_compat = QtWidgets.QWidget()
        foc = QtWidgets.QFormLayout(pane_oai_compat)
        oai_compat_base_le = QtWidgets.QLineEdit()
        oai_compat_base_le.setPlaceholderText('http://localhost:1234/v1')
        oai_compat_key_le = QtWidgets.QLineEdit()
        oai_compat_key_le.setEchoMode(QtWidgets.QLineEdit.Password)
        oai_compat_model_le = QtWidgets.QComboBox()
        oai_compat_model_le.setEditable(True)
        btn_oai_compat_refresh = QtWidgets.QPushButton('Refresh Models')
        btn_oai_compat_test = QtWidgets.QPushButton('Test OpenAI-Compatible')
        lm_tip = QtWidgets.QLabel('Tip (LLMStudio/LM Studio): Start the OpenAI-compatible server on 127.0.0.1:1234, use base URL http://127.0.0.1:1234/v1, leave API key empty unless configured.')
        lm_tip.setWordWrap(True)
        foc.addRow('Base URL:', oai_compat_base_le)
        foc.addRow('API key (optional):', oai_compat_key_le)
        foc.addRow('Model:', oai_compat_model_le)
        foc.addRow(btn_oai_compat_refresh, btn_oai_compat_test)
        foc.addRow(lm_tip)
        stack.addWidget(pane_oai_compat)

        vbox.addWidget(stack)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        vbox.addWidget(btns)

        # Load current settings
        s = QSettings('Pistonspy', 'TagOmatic')
        use_cloud_cb.setChecked(bool(s.value('cloud/use_cloud', False, type=bool)))
        provider_combo.setCurrentText(s.value('cloud/preferred_provider', 'auto', type=str))
        ollama_host_le.setText(s.value('cloud/ollama_host', 'http://localhost:11434', type=str))
        oai_compat_base_le.setText(s.value('cloud/oai_compat_base_url', 'http://localhost:1234/v1', type=str))
        oai_compat_key_le.setText(s.value('cloud/oai_compat_api_key', '', type=str))
        saved_oai_compat_model = s.value('cloud/oai_compat_model', '', type=str)
        if saved_oai_compat_model and saved_oai_compat_model not in [oai_compat_model_le.itemText(i) for i in range(oai_compat_model_le.count())]:
            oai_compat_model_le.addItem(saved_oai_compat_model)
        oai_compat_model_le.setCurrentText(saved_oai_compat_model)

        def on_provider_changed(idx: int):
            # Map provider combo to stack index
            # Stack order: 0=Remote Ollama, 1=OpenAI-compatible
            provider_text = provider_combo.currentText().lower()
            if provider_text == 'remote-ollama':
                stack.setCurrentIndex(0)
            elif provider_text == 'openai-compatible':
                stack.setCurrentIndex(1)
            else:  # auto or unknown
                stack.setCurrentIndex(0)  # Default to Remote Ollama pane
        provider_combo.currentIndexChanged.connect(on_provider_changed)
        on_provider_changed(provider_combo.currentIndex())

        def refresh_ollama():
            try:
                import urllib.request, json
                base = ollama_host_le.text().strip().rstrip('/')
                url = base + '/api/tags'
                data = urllib.request.urlopen(url, timeout=2).read().decode('utf-8', 'ignore')
                obj = json.loads(data)
                names = []
                try:
                    for it in (obj.get('models') or []):
                        n = str(it.get('name','')).strip()
                        if n:
                            names.append(n)
                except Exception:
                    pass
                if names:
                    ollama_model_le.clear()
                    ollama_model_le.addItems(names)
                    QtWidgets.QMessageBox.information(dlg, 'Ollama', f'Found {len(names)} models')
                else:
                    QtWidgets.QMessageBox.warning(dlg, 'Ollama', 'No models found')
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, 'Ollama', f'Failed: {e}')
        btn_ollama_refresh.clicked.connect(refresh_ollama)

        def test_ollama():
            try:
                import urllib.request
                base = ollama_host_le.text().strip().rstrip('/')
                urllib.request.urlopen(base + '/api/version', timeout=2).read(1)
                QtWidgets.QMessageBox.information(dlg, 'Ollama', 'Connection OK')
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, 'Ollama', f'Failed: {e}')
        btn_ollama_test.clicked.connect(test_ollama)

        def refresh_oai_compat():
            try:
                import urllib.request, json
                base = (oai_compat_base_le.text().strip() or 'http://localhost:1234/v1').rstrip('/')
                url = base + '/models'
                headers = {}
                if oai_compat_key_le.text().strip():
                    headers['Authorization'] = f"Bearer {oai_compat_key_le.text().strip()}"
                req = urllib.request.Request(url, headers=headers)
                data = urllib.request.urlopen(req, timeout=5).read().decode('utf-8', 'ignore')
                obj = json.loads(data)
                names = []
                try:
                    for it in (obj.get('data') or []):
                        n = str(it.get('id','')).strip()
                        if n:
                            names.append(n)
                except Exception:
                    pass
                if names:
                    oai_compat_model_le.clear()
                    oai_compat_model_le.addItems(names)
                    QtWidgets.QMessageBox.information(dlg, 'OpenAI-Compatible', f'Found {len(names)} models')
                else:
                    QtWidgets.QMessageBox.warning(dlg, 'OpenAI-Compatible', 'No models found')
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, 'OpenAI-Compatible', f'Failed: {e}')
        btn_oai_compat_refresh.clicked.connect(refresh_oai_compat)

        def test_oai_compat():
            try:
                import urllib.request
                base = (oai_compat_base_le.text().strip() or 'http://localhost:1234/v1').rstrip('/')
                urllib.request.urlopen(base + '/models', timeout=5).read(1)
                QtWidgets.QMessageBox.information(dlg, 'OpenAI-Compatible', 'Connection OK')
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, 'OpenAI-Compatible', f'Failed: {e}')
        btn_oai_compat_test.clicked.connect(test_oai_compat)

        def on_save():
            # Persist core flags
            s.setValue('cloud/use_cloud', use_cloud_cb.isChecked())
            s.setValue('cloud/preferred_provider', provider_combo.currentText())
            # External Ollama
            s.setValue('cloud/ollama_host', ollama_host_le.text().strip())
            s.setValue('cloud/ollama_model', ollama_model_le.currentText().strip())
            # OpenAI-compatible
            s.setValue('cloud/oai_compat_base_url', oai_compat_base_le.text().strip())
            s.setValue('cloud/oai_compat_api_key', oai_compat_key_le.text().strip())
            s.setValue('cloud/oai_compat_model', oai_compat_model_le.currentText().strip())
            # Also persist preferred provider
            s.setValue('cloud/preferred_provider', provider_combo.currentText())
            s.sync()
            try:
                cloud_router.configure(
                    preferred_provider=provider_combo.currentText(),
                    use_cloud=use_cloud_cb.isChecked(),
                    external_ollama_host=s.value('cloud/ollama_host', 'http://localhost:11434', type=str),
                    external_ollama_model=s.value('cloud/ollama_model', '', type=str),
                    oai_compat_base_url=s.value('cloud/oai_compat_base_url', 'http://localhost:1234/v1', type=str),
                    oai_compat_api_key=s.value('cloud/oai_compat_api_key', '', type=str),
                    oai_compat_model=s.value('cloud/oai_compat_model', '', type=str),
                )
                # Immediately refresh connection label and disable local selectors when cloud active
                try:
                    self._update_connection_label()
                except Exception:
                    pass
            except Exception:
                pass
            dlg.accept()
        btns.accepted.connect(on_save)
        btns.rejected.connect(dlg.reject)
        dlg.exec()

    def check_connection(self):
        # In cloud mode, treat provider as authoritative and do not show local disconnect
        try:
            if getattr(cloud_router, 'use_cloud', False):
                prov = getattr(cloud_router, 'preferred_provider', '') or ''
                if (cloud_router.openai_compat.is_configured() or cloud_router.remote_ollama.is_configured() or prov in {'openai-compatible', 'remote-ollama', 'auto'}):
                    self._update_connection_label()
                    return True
        except Exception:
            pass
        # Otherwise check local Ollama
        try:
            info = self.client.show(model=self.model_name)
            if info is not None:
                self.connection_label.setText(f'🟢 Connected - {self.model_name}')
                return True
        except Exception:
            pass
        self.connection_label.setText('🔴 Disconnected')
        return False

    def warm_model_async(self):
        """Non-blocking warmup using a Python thread to avoid Qt thread lifetime issues.
        Uses a longer timeout to tolerate model load (esp. first run).
        NOW WARMS BOTH MODELS to prevent switching delays!"""
        def _do_warm():
            try:
                # Use a longer-timeout client just for warmup
                warm_client = ollama.Client(host='http://localhost:11434', timeout=300)
                
                # Warm up MAIN model
                try:
                    warm_client.show(model=self.model_name)
                except Exception:
                    pass
                # Minimal chat to trigger load & keep-alive
                warm_client.chat(model=self.model_name, messages=[{'role': 'user', 'content': 'warmup'}], keep_alive='30m')
                
                # CRITICAL: Warm up NON-CAR model to prevent 31+ second delays!
                try:
                    noncar_model = getattr(self, 'noncar_model', None)
                            # CRITICAL FIX: No more model warming - causes 31+ second delays
        # CRITICAL FIX: No more model warming - causes 31+ second delays
        # if noncar_model and noncar_model != self.model_name:
        #     print(f"[DEBUG] Warming non-car model: {noncar_model}")
        #     warm_client.show(model=noncar_model)
        #     warm_client.chat(model=noncar_model, messages=[{'role': 'user', 'content': 'warmup'}], keep_alive='30m')
        #     print(f"[DEBUG] Non-car model warmed successfully")
        # else:
        #     print(f"[DEBUG] No separate non-car model to warm")
                except Exception as e:
                    print(f"[DEBUG] Non-car model warmup failed (non-critical): {e}")
                
                QtCore.QMetaObject.invokeMethod(self, '_on_warm_ok', Qt.QueuedConnection)
            except Exception as e:
                # Pass error string back to UI thread but do not block use
                self._last_warm_err = str(e)
                QtCore.QMetaObject.invokeMethod(self, '_on_warm_err', Qt.QueuedConnection)
        th = threading.Thread(target=_do_warm, daemon=True)
        th.start()

    @QtCore.Slot()
    def _on_warm_ok(self):
        try:
            self._update_connection_label()
        except Exception:
            self.connection_label.setText(f'🟢 Connected - {self.model_name}')
        self.status.showMessage('Model warmed up', 3000)

    @QtCore.Slot()
    def _on_warm_err(self):
        err = getattr(self, '_last_warm_err', 'unknown error')
        self.status.showMessage(f'Warmup timed out/failed (continuing): {err}', 6000)

    def _check_latest_version_async(self):
        print(f"[DEBUG] Version check method called!")
        def _do():
            print(f"[DEBUG] Version check thread started")
            try:
                print(f"[DEBUG] Checking URL: {VERSION_CHECK_URL}")
                import urllib.request
                import urllib.error
                import socket
                req = urllib.request.Request(VERSION_CHECK_URL, headers={'User-Agent': 'TagOmatic-Client'})
                print(f"[DEBUG] Making HTTP request...")
                with urllib.request.urlopen(req, timeout=10) as resp:  # Increased timeout to 10 seconds
                    raw = resp.read().decode('utf-8', errors='ignore')
                print(f"[DEBUG] Response received: {raw[:100]}...")
                latest = None
                download_url = None
                news = None
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        latest = str(obj.get('latest', '')).strip()
                        download_url = str(obj.get('download_url', '')).strip()
                        news = str(obj.get('news', '')).strip()
                        print(f"[DEBUG] Parsed JSON: latest={latest}, download_url={download_url}, news={news}")
                except Exception as e:
                    print(f"[DEBUG] JSON parse error: {e}")
                    # Accept plain string like {"latest":"v1.1f"} or even just v1.1f text
                    if 'v' in raw:
                        latest = raw.strip().split()[0]
                        print(f"[DEBUG] Fallback parsing: latest={latest}")
                
                print(f"[DEBUG] Final comparison: Current={APP_VERSION}, Latest={latest}")
                if latest and latest != APP_VERSION:
                    print(f"[DEBUG] Version mismatch detected! Creating popup...")
                    
                    def _open_download():
                        """Helper function to open download URL"""
                        try:
                            print(f"[DEBUG] Opening download URL: {download_url}")
                            import webbrowser
                            webbrowser.open(download_url)
                            print(f"[DEBUG] Download URL opened in browser")
                        except Exception as e:
                            print(f"[DEBUG] Error opening download URL: {e}")
                            # Fallback to QDesktopServices
                            try:
                                QtGui.QDesktopServices.openUrl(QtCore.QUrl(download_url))
                                print(f"[DEBUG] Fallback to QDesktopServices")
                            except Exception as e2:
                                print(f"[DEBUG] QDesktopServices also failed: {e2}")
                    
                    def _open_website():
                        """Helper function to open website"""
                        try:
                            print(f"[DEBUG] Opening website")
                            import webbrowser
                            webbrowser.open('https://www.tagomatic.co.uk/')
                            print(f"[DEBUG] Website opened in browser")
                        except Exception as e:
                            print(f"[DEBUG] Error opening website: {e}")
                            # Fallback to QDesktopServices
                            try:
                                QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://www.tagomatic.co.uk/'))
                                print(f"[DEBUG] Fallback to QDesktopServices")
                            except Exception as e2:
                                print(f"[DEBUG] QDesktopServices also failed: {e2}")
                    
                    # Use a simple QMessageBox instead of custom dialog
                    def _show_simple_popup():
                        try:
                            print(f"[DEBUG] Creating simple QMessageBox popup...")
                            
                            # Create a simple message box
                            msg = QtWidgets.QMessageBox(self)
                            msg.setWindowTitle('New Version Available! 🚀')
                            msg.setText(f'New version available: {latest}\n\nYou are running: {APP_VERSION}')
                            
                            if news:
                                msg.setInformativeText(f'What\'s New:\n{news}')
                            
                            # Add buttons
                            if download_url:
                                download_btn = msg.addButton('Download New Version', QtWidgets.QMessageBox.ActionRole)
                                download_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(download_url)))
                            
                            website_btn = msg.addButton('Visit Website', QtWidgets.QMessageBox.ActionRole)
                            website_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://www.tagomatic.co.uk/')))
                            

                            
                            close_btn = msg.addButton('Close', QtWidgets.QMessageBox.RejectRole)
                            
                            print(f"[DEBUG] Simple popup created, showing...")
                            result = msg.exec()
                            print(f"[DEBUG] Simple popup result: {result}")
                            
                        except Exception as e:
                            print(f"[DEBUG] Error creating simple popup: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Schedule the simple popup
                    QtCore.QTimer.singleShot(100, _show_simple_popup)
                        
                elif latest == APP_VERSION:
                    # Version matches, no popup needed
                    pass
                    
            except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
                print(f"[DEBUG] Version check timeout/connection error (silently ignored): {e}")
                # Connection failed or timeout - skip silently, don't interrupt user
                pass
            except Exception as e:
                print(f"[DEBUG] Version check error (silently ignored): {e}")
                # Any other error - skip silently
                pass
                
        # Use QTimer instead of threading to ensure UI updates happen on main thread
        QtCore.QTimer.singleShot(100, _do)
    def refresh_models(self):
        try:
            names = list_models_via_http(self.client)
            if not names:
                self.model_combo.clear()
                self.status.showMessage('No models found', 5000)
                return
            self.model_combo.clear()
            self.model_combo.addItems(names)
            current = self.model_name
            if current and any(_names_match(current, n) for n in names):
                picked = next((n for n in names if _names_match(current, n)), names[0])
            else:
                picked = pick_preferred_vision_model(names, self.model_name)
            self.model_combo.setCurrentText(picked)
            self.model_name = picked
            self.status.showMessage(f'Found {len(names)} models', 5000)
            # Confirm connection and warm selected model
            self.check_connection()
            self.warm_model_async()
        except Exception as e:
            self.status.showMessage(f'Error refreshing models: {e}', 8000)

    @Slot(str)
    def on_model_changed(self, text: str):
        if text:
            # Ignore placeholder in cloud mode
            chosen = text.strip()
            if chosen and chosen != '(None - Cloud)':
                # Unload previous model if it's different and we're using local Ollama
                try:
                    if getattr(cloud_router, 'use_cloud', False):
                        # Cloud mode - skip unloading
                        pass
                    else:
                        # Local Ollama mode - unload previous model(s) if different
                        # Unload the currently active model if it's different from the new one
                        if self.model_name and self.model_name != chosen:
                            self._unload_model(self.model_name)
                        # Also unload any previously tracked model if it's different
                        if self.previous_model_name and self.previous_model_name != chosen and self.previous_model_name != self.model_name:
                            self._unload_model(self.previous_model_name)
                except Exception as e:
                    _log(f"Error unloading previous model: {e}")
                
                # Update model tracking
                self.previous_model_name = self.model_name if self.model_name != chosen else None
                self.model_name = chosen
            # If cloud active, update label and skip local warmup
            try:
                if getattr(cloud_router, 'use_cloud', False) and (getattr(cloud_router, 'openai_api_key', '') or getattr(cloud_router, 'gemini_api_key', '')):
                    self._update_connection_label()
                    return
            except Exception:
                pass
            self.check_connection()
            self.warm_model_async()
    
    def _unload_model(self, model_name: str):
        """Unload a model from Ollama memory using the API
        
        This runs in a background thread to avoid blocking the UI.
        Uses Ollama's keep_alive=0 parameter to force model unloading.
        """
        def _unload_in_thread():
            try:
                _log(f"Unloading previous model: {model_name}")
                # Use Ollama's API to unload the model by making a minimal generate request with keep_alive=0
                # This forces Ollama to unload the model from memory after processing
                ollama_host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
                if not ollama_host.startswith('http'):
                    ollama_host = f'http://{ollama_host}'
                
                # Make a minimal generate request with keep_alive=0 to unload the model
                # The model will be unloaded immediately after this request completes
                response = requests.post(
                    f'{ollama_host}/api/generate',
                    json={
                        'model': model_name,
                        'prompt': '.',  # Minimal prompt - just a dot to trigger processing
                        'keep_alive': '0'  # Unload immediately after response
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    _log(f"Successfully unloaded model: {model_name}")
                else:
                    _log(f"Warning: Failed to unload model {model_name}, status: {response.status_code}")
            except Exception as e:
                # Don't fail if unloading doesn't work - it's just a memory optimization
                _log(f"Could not unload model {model_name}: {e}")
        
        # Run unload in background thread to avoid blocking UI
        threading.Thread(target=_unload_in_thread, daemon=True).start()

    def select_and_process_image(self):
        dlg = QtWidgets.QFileDialog(self, 'Select Image')
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        # Support all common image formats that exiftool can write to
        dlg.setNameFilters([
            'All Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif *.webp *.heic *.heif *.cr2 *.nef *.arw *.raf *.orf *.rw2 *.dng *.raw)',
            'JPEG (*.jpg *.jpeg)',
            'RAW (*.cr2 *.nef *.arw *.raf *.orf *.rw2 *.dng *.raw)',
            'Other Formats (*.png *.bmp *.tif *.tiff *.gif *.webp *.heic *.heif)',
            'All Files (*.*)'
        ])
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            path = dlg.selectedFiles()[0]
            self.process_image(path)

    def select_and_process_folders(self):
        folders = self._select_multiple_directories()
        if folders:
            self.batch_process_multiple_folders(folders)

    def _select_multiple_directories(self) -> list[str]:
        """Open a non-native dialog to allow multi-select of folders (Qt limitation)."""
        try:
            dlg = QtWidgets.QFileDialog(self, 'Select Folders')
            dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)
            dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            dlg.setOption(QtWidgets.QFileDialog.DontResolveSymlinks, True)
            try:
                if getattr(self, '_last_dir', None) and os.path.isdir(self._last_dir):
                    dlg.setDirectory(self._last_dir)
            except Exception:
                pass
            # Allow multi-selection on both views; we will filter and de-parent after
            for view in dlg.findChildren(QtWidgets.QListView) + dlg.findChildren(QtWidgets.QTreeView):
                try:
                    view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
                except Exception:
                    pass
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                # Try robust extraction from internal views via QFileSystemModel
                selected_dirs = set()
                try:
                    views = dlg.findChildren(QtWidgets.QListView) + dlg.findChildren(QtWidgets.QTreeView)
                    for v in views:
                        try:
                            sel = v.selectionModel().selectedIndexes()
                        except Exception:
                            sel = []
                        model = getattr(v, 'model', lambda: None)()
                        for idx in sel:
                            try:
                                # Only column 0 entries represent unique items
                                if getattr(idx, 'column', lambda: 0)() != 0:
                                    continue
                                filePath = getattr(model, 'filePath', None)
                                isDir = getattr(model, 'isDir', None)
                                path = filePath(idx) if callable(filePath) else ''
                                if path and os.path.isdir(path):
                                    # If model exposes isDir, prefer that
                                    if callable(isDir):
                                        try:
                                            if not isDir(idx):
                                                continue
                                        except Exception:
                                            pass
                                    selected_dirs.add(path)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Fallback to selectedFiles if views extraction empty (ignore footer "Directory:" text)
                if not selected_dirs:
                    try:
                        picked = dlg.selectedFiles()
                        for p in picked:
                            # Filter out any concatenated footer string (contains quotes and spaces)
                            if '"' in p and not os.path.exists(p):
                                continue
                            if os.path.isdir(p):
                                selected_dirs.add(p)
                    except Exception:
                        pass
                # If still empty, try the currentIndex from any view (single selection by focus)
                if not selected_dirs:
                    try:
                        views = dlg.findChildren(QtWidgets.QListView) + dlg.findChildren(QtWidgets.QTreeView)
                        for v in views:
                            idx = v.currentIndex()
                            if idx and getattr(idx, 'isValid', lambda: False)():
                                if getattr(idx, 'column', lambda: 0)() != 0:
                                    idx = idx.sibling(idx.row(), 0)
                                model = getattr(v, 'model', lambda: None)()
                                filePath = getattr(model, 'filePath', None)
                                isDir = getattr(model, 'isDir', None)
                                path = filePath(idx) if callable(filePath) else ''
                                if path and os.path.isdir(path):
                                    if callable(isDir):
                                        try:
                                            if not isDir(idx):
                                                continue
                                        except Exception:
                                            pass
                                    selected_dirs.add(path)
                    except Exception:
                        pass
                # If still empty (e.g., user double-clicked into a folder and hit Choose), use current directory
                if not selected_dirs:
                    try:
                        cur = dlg.directory().absolutePath()
                        if cur and os.path.isdir(cur):
                            selected_dirs.add(cur)
                    except Exception:
                        pass
                # Remove parent directories when a deeper child is also selected (compare on normalized paths, but return originals)
                try:
                    # Normalize to avoid case/sep differences on Windows
                    def _norm(p: str) -> str:
                        try:
                            return os.path.normcase(os.path.normpath(str(Path(p))))
                        except Exception:
                            return os.path.normcase(os.path.normpath(p))
                    origs = list(selected_dirs)
                    pairs = [(o, _norm(o)) for o in origs]
                    # Sort by normalized depth/length desc
                    pairs.sort(key=lambda t: (len(t[1]), t[1]), reverse=True)
                    kept_orig: list[str] = []
                    kept_norm: list[str] = []
                    for o, n in pairs:
                        if any(os.path.commonpath([n, kn]).lower() == kn.lower() for kn in kept_norm):
                            continue
                        kept_orig.append(o)
                        kept_norm.append(n)
                    selected_dirs = set(kept_orig)
                    # Drop the dialog's current directory root if deeper children exist under it (compare on normalized)
                    try:
                        root_o = dlg.directory().absolutePath()
                        root_n = _norm(root_o)
                        if len(kept_norm) > 1 and any(os.path.commonpath([n, root_n]).lower() == root_n.lower() and o != root_o for o, n in zip(kept_orig, kept_norm)):
                            # remove root_o from originals if present
                            if root_o in selected_dirs:
                                selected_dirs.discard(root_o)
                    except Exception:
                        pass
                except Exception:
                    pass
                # Update last dir for next time
                try:
                    if selected_dirs:
                        first = sorted(selected_dirs)[0]
                        self._last_dir = first
                except Exception:
                    pass
                return sorted(selected_dirs)
        except Exception:
            pass
        return []

    def batch_process_multiple_folders(self, folders: list[str]):
        """Start a streaming batch over multiple base folders."""
        # Support all common image formats that exiftool can handle
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp', '.heic', '.heif',
                '.cr2', '.nef', '.arw', '.raf', '.orf', '.rw2', '.dng', '.raw', '.tiff'}
        # Normalize and verify folder list
        try:
            folders = [str(Path(f)) for f in folders if os.path.isdir(f)]
        except Exception:
            pass
        if not folders:
            self.status.showMessage('No valid folders selected', 4000)
            return
        self.is_batch = True
        self._batch_results = []  # Clear batch results for new batch
        self.batch_total = 0
        self.batch_done = 0
        self.batch_failed = 0
        self.batch_skipped = 0
        self.dynamic_limit = self.max_concurrent  # Use user's setting directly
        # Ensure paused state is reset when starting new batch
        self.paused = False
        self.btn_pause.setText('Pause')
        self.success_streak = 0
        self.failure_streak = 0
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self.status.showMessage('Starting multi-folder batch (streaming)...')
        self._console_log(f"[BATCH] multi-folder window={self.batch_window_size} concurrency={self.max_concurrent}")
        self._pending_queue.clear()
        self._batch_remaining = []
        def _iter_files_multi():
            for base in folders:
                try:
                    base_path = Path(base)
                    if self.recursive_checkbox.isChecked():
                        for root, _dirs, files_local in os.walk(base_path):
                            for name in files_local:
                                if Path(name).suffix.lower() in exts:
                                    yield str(Path(root) / name)
                    else:
                        for p in base_path.glob('*'):
                            if p.suffix.lower() in exts:
                                yield str(p)
                except Exception:
                    pass
        self._dir_iter = _iter_files_multi()
        self._refill_pending_window()
        self._drain_queue()
        # If nothing was queued (e.g., empty selection), finalize gracefully
        try:
            if self.batch_total == 0 and not self._pending_queue and not self.inflight:
                self.is_batch = False
                self.progress.setVisible(False)
                self.progress.setRange(0, 1)
                QtWidgets.QMessageBox.information(self, 'Batch Complete', 'No images found in selected folders.')
        except Exception:
            pass

    def process_image(self, path: str):
        # If paused, queue for later and return immediately
        if getattr(self, 'paused', False):
            try:
                # Avoid duplicates in queue
                if path not in self._pending_queue and path not in self.inflight:
                    self._pending_queue.append(path)
                self.status.showMessage(f'Paused: queued {os.path.basename(path)}', 3000)
            except Exception:
                pass
            return
        # Respect overwrite policy before scheduling
        policy = self.overwrite_combo.currentText().strip().lower()
        # In batch mode, perform async exiftool metadata check to avoid UI stalls
        if self.is_batch and policy != 'overwrite':
            if path in self._active_meta:
                return
            task = MetaCheckTask(path)
            worker = MetaRunnable(task)
            task.finished.connect(self.on_meta_checked)
            self._active_meta[path] = (task, worker)
            # Run metadata checks in dedicated, throttled pool
            self.meta_pool.start(worker)
            return
        # Non-batch: synchronous check keeps existing behavior
        # NOTE: We now ignore existing metadata during inference for cleaner processing
        if policy != 'overwrite' and self._metadata_exists(path, fast_only=False):
            if self.is_batch:
                # In batch, avoid blocking; default to skip
                self.batch_skipped += 1
                self.status.showMessage(f'Skipped (metadata exists): {os.path.basename(path)}', 3000)
                self._console_log(f"[SKIP] existing metadata: {path}")
                # Update progress
                self.progress.setVisible(True)
                self.progress.setRange(0, self.batch_total)
                self.progress.setValue(self.batch_done + self.batch_failed + self.batch_skipped)
                # Refill next items since this one was skipped
                self._refill_pending_window()
                self._check_batch_completion_and_finish()
                return
            if policy == 'ask':
                res = QtWidgets.QMessageBox.question(self, 'Existing Metadata',
                    f'This image already has metadata. Overwrite?\n\nYes = Overwrite\nNo = Skip',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No)
                if res != QtWidgets.QMessageBox.Yes:
                    self.status.showMessage(f'Skipped: {os.path.basename(path)}', 3000)
                    self._console_log(f"[SKIP] user chose not to overwrite: {path}")
                    return
            else:
                self.status.showMessage(f'Skipped (metadata exists): {os.path.basename(path)}', 3000)
                self._console_log(f"[SKIP] existing metadata (single): {path}")
                return
        self._process_image_core(path)

    def _process_image_core(self, path: str):
        # Dynamic concurrency cap
        effective_limit = min(self.max_concurrent, self.dynamic_limit)
        if len(self.inflight) >= effective_limit:
            # Queue for later to avoid creating thousands of timers in big batches
            try:
                if path not in self._pending_queue and path not in self.inflight:
                    self._pending_queue.append(path)
            except Exception:
                pass
            return
        self.model_name = self.model_combo.currentText().strip() or self.model_name
        if not self.model_name:
            self.status.showMessage('No model selected', 4000)
            return
        task = InferenceTask(self.client, self.model_name, path, self.embed_checkbox.isChecked())
        # Pass motorshow mode from UI
        try:
            task.motorshow_mode = bool(self.motorshow_checkbox.isChecked())
        except Exception:
            task.motorshow_mode = False
        # Auto-enable race mode if race tagging is enabled and PDF is loaded
        task.race_mode = bool(self.race_tagging_enabled and self.race_data)
        # Motorshow disabled checkbox removed - always set to False
        task.motorshow_disabled = False
        worker = TaskRunnable(task)
        task.finished.connect(self.on_task_finished)
        task.failed.connect(self.on_task_failed)
        self.inflight.add(path)
        self._active_tasks[path] = (task, worker)
        self._console_log(f"[START] {path}")
        self.threadpool.start(worker)
        self.status.showMessage(f'Processing: {os.path.basename(path)}')
        # Update inference mode indicator
        try:
            if hasattr(task, 'motorshow_mode') and task.motorshow_mode:
                self.inference_mode_label.setText('Mode: Motorshow')
                self.inference_mode_label.setStyleSheet('QLabel { background-color: #0c1a33; border: 1px solid #2563eb; border-radius: 10px; padding: 2px 10px; color: #87ceeb; font-weight: bold; }')
            else:
                self.inference_mode_label.setText('Mode: Car Analysis')
                self.inference_mode_label.setStyleSheet('QLabel { background-color: #0c1a33; border: 1px solid #2563eb; border-radius: 10px; padding: 2px 10px; color: #87ceeb; font-weight: bold; }')
        except Exception:
            pass
        # Show activity indicator
        try:
            self._set_activity(True)
        except Exception:
            pass
        # Show progress for single-item mode
        if not self.is_batch:
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # busy/indeterminate
        # Assign processing slot and record start
        self._assign_processing_slot(path)
        self.task_started_at[path] = time.time()
        # Update progress text for large batches
        if self.is_batch:
            self.status.showMessage(f'Queued/In-flight: {self.batch_done + self.batch_failed + len(self.inflight)}/{self.batch_total or 0}')

    @Slot(str, bool)
    def on_meta_checked(self, image_path: str, exists: bool):
        self._active_meta.pop(image_path, None)
        if exists:
            # Count as skipped in batch mode
            if self.is_batch:
                self.batch_skipped += 1
                self.progress.setVisible(True)
                if self.batch_total:
                    self.progress.setRange(0, self.batch_total)
                self.progress.setValue(self.batch_done + self.batch_failed + self.batch_skipped)
                self.status.showMessage(f'Skipped (metadata exists): {os.path.basename(image_path)}', 2000)
                self._console_log(f"[SKIP] existing metadata (async): {image_path}")
                self._refill_pending_window()
                self._check_batch_completion_and_finish()
            return
        # Otherwise proceed to process now
        self._process_image_core(image_path)

    @Slot(str, dict, str)
    def on_task_finished(self, image_path: str, parsed: dict, raw: str):
        # Ignore if already timed out and dropped
        if image_path in self.timed_out:
            self._clear_processing_slot(image_path)
            self._active_tasks.pop(image_path, None)
            return
        # Update rolling timing
        started = self.task_started_at.pop(image_path, None)
        if isinstance(started, (int, float)):
            self._record_duration(time.time() - started)
        self.inflight.discard(image_path)
        self._active_tasks.pop(image_path, None)
        self._clear_processing_slot(image_path)
        
        # VALIDATION CHECK: Validate Make/Model against knowledge base
        validation_retry_count = self._validation_retries.get(image_path, 0)
        if self.vehicle_kb_path and validation_retry_count < self._max_validation_retries:
            is_valid, validation_notes, model_data = validate_vehicle_identification(parsed, self.vehicle_kb_path)
            
            if not is_valid:
                # Validation failed - auto-retry with JSON context
                self._validation_retries[image_path] = validation_retry_count + 1
                _log(f"[VALIDATION-FAIL] {image_path}: {validation_notes} (retry {validation_retry_count + 1}/{self._max_validation_retries})")
                
                # Build feedback context with JSON distinguishing_cues
                feedback_comments = f"Validation failed: {validation_notes}\n\n"
                if model_data:
                    # If we found a partial match or similar model, include its distinguishing cues
                    cues = model_data.get('distinguishing_cues', {})
                    if cues:
                        feedback_comments += "Reference distinguishing features from knowledge base:\n"
                        for angle, desc in cues.items():
                            if desc:
                                feedback_comments += f"- {angle.capitalize()}: {desc}\n"
                else:
                    # No match found - suggest checking available models
                    make = parsed.get('Make', 'Unknown')
                    kb = load_vehicle_knowledge_base(self.vehicle_kb_path)
                    approved_kb = load_approved_knowledge_base()
                    user_kb = load_user_knowledge_base()
                    merged_kb = merge_knowledge_bases(kb, approved_kb, user_kb)
                    if make in merged_kb:
                        available = [m.get('model', '') for m in merged_kb[make][:5]]
                        feedback_comments += f"Available models for {make}: {', '.join(available)}\n"
                
                # Auto-retry with feedback context
                self._console_log(f"[VALIDATION-RETRY] {image_path}: {validation_notes}")
                task = InferenceTask(self.client, self.model_name, image_path, self.embed_checkbox.isChecked())
                task.feedback_context = {
                    'reason': 'Validation failed - Make/Model not found in knowledge base',
                    'comments': feedback_comments
                }
                try:
                    task.motorshow_mode = bool(self.motorshow_checkbox.isChecked())
                except Exception:
                    task.motorshow_mode = False
                # Auto-enable race mode if race tagging is enabled and PDF is loaded
                task.race_mode = bool(self.race_tagging_enabled and self.race_data)
                worker = TaskRunnable(task)
                task.finished.connect(self.on_task_finished)
                task.failed.connect(self.on_task_failed)
                self.inflight.add(image_path)
                self._active_tasks[image_path] = (task, worker)
                self.task_started_at[image_path] = time.time()
                self._assign_processing_slot(image_path)
                self.threadpool.start(worker)
                self.status.showMessage(f"Validation failed - retrying with KB context: {os.path.basename(image_path)}", 5000)
                return  # Don't display result yet - wait for retry
        
        # Validation passed or max retries reached - display result
        # Reset retry count on success
        if image_path in self._validation_retries:
            del self._validation_retries[image_path]
        
        # RACE METADATA TAGGING: Apply race metadata from PDF if enabled
        _log(f"[RACE-TAG] Checking conditions: race_tagging_enabled={self.race_tagging_enabled}, race_data={bool(self.race_data)}, race_sessions={bool(self.race_sessions)}")
        if self.race_tagging_enabled and self.race_data and self.race_sessions:
            try:
                race_number = parsed.get('Race Number', '').strip()
                _log(f"[RACE-TAG] Raw race number from parsed dict: '{race_number}' (type: {type(race_number)})")
                # Clean race number - remove non-numeric characters except for multi-digit numbers
                # Handle cases like "79", "(79)", "Race 79", etc.
                if race_number:
                    import re
                    # Extract just the number part
                    num_match = re.search(r'\d+', race_number)
                    if num_match:
                        race_number = num_match.group(0)
                    else:
                        race_number = ''  # Invalid race number
                
                original_race_num = parsed.get('Race Number', '')
                _log(f"[RACE-TAG] Checking race number for {os.path.basename(image_path)}: extracted='{original_race_num}', cleaned='{race_number}', PDF has {len(self.race_data)} entries")
                
                # Add track and event name from PDF (global for all images from this PDF)
                if self.race_track:
                    parsed['Race Track'] = self.race_track
                if self.race_event_name:
                    parsed['Event Name'] = self.race_event_name
                
                if race_number and race_number != 'Unknown':
                    # Determine active session (PRIMARY: timestamp, OVERRIDE: manual)
                    active_session = self.get_active_session(image_path)
                    if active_session:
                        _log(f"[RACE-TAG] Using session: {active_session} (from {'manual selection' if self.selected_session else 'EXIF timestamp'})")
                    
                    # Filter race data by session
                    filtered_data = self.get_filtered_race_data(active_session)
                    
                    # Get AI-identified Make/Model/Color for car matching
                    ai_make = parsed.get('Make', '').strip()
                    ai_model = parsed.get('Model', '').strip()
                    ai_color = parsed.get('Color', '').strip()
                    
                    # Find all matches for race number in filtered data
                    all_matches = []
                    race_number_lower = race_number.lower()
                    for num, data in filtered_data.items():
                        if num.lower() == race_number_lower:
                            all_matches.append((num, data))
                    
                    # Match race number to PDF data
                    if len(all_matches) > 1:
                        # Multiple matches - use car matching
                        _log(f"[RACE-TAG] Multiple matches for #{race_number}, using car matching")
                        matched_data = match_race_number_with_car(race_number, ai_make, ai_model, ai_color, filtered_data)
                        if matched_data:
                            _log(f"[RACE-TAG] Matched via car matching: Make={ai_make}, Model={ai_model}, Color={ai_color}")
                    else:
                        # Single or no match - use standard matching
                        matched_data = match_race_number_to_data(race_number, filtered_data)
                        if not matched_data and active_session:
                            # Fallback to all sessions if no match in filtered session
                            _log(f"[RACE-TAG] No match in session {active_session}, trying all sessions")
                            matched_data = match_race_number_to_data(race_number, self.race_data)
                    
                    if matched_data:
                        _log(f"[RACE-TAG] Matched race number {race_number} to PDF data: {matched_data}")
                        # PDF is source of truth - override AI Make/Model with PDF data
                        if matched_data.get('car'):
                            pdf_car = matched_data['car'].strip()
                            # Smart parsing: Look for common car makes in the string
                            # Common car makes (case-insensitive matching)
                            common_makes = [
                                'Nissan', 'Suzuki', 'Toyota', 'Honda', 'Mazda', 'Subaru', 'Mitsubishi',
                                'BMW', 'Mercedes', 'Audi', 'Porsche', 'Volkswagen', 'Volvo', 'Mini',
                                'Ford', 'Chevrolet', 'Dodge', 'Jeep', 'Chrysler', 'Cadillac', 'Buick',
                                'Ferrari', 'Lamborghini', 'McLaren', 'Aston Martin', 'Bentley', 'Rolls-Royce',
                                'Jaguar', 'Land Rover', 'Range Rover', 'Alfa Romeo', 'Fiat', 'Peugeot',
                                'Renault', 'Citroen', 'MG', 'TVR', 'Lotus', 'Caterham', 'Noble',
                                'Pagani', 'Koenigsegg', 'Bugatti', 'Ariel', 'BAC', 'Ginetta'
                            ]
                            
                            # Try to find a car make in the string
                            found_make = None
                            make_start_idx = None
                            pdf_car_upper = pdf_car.upper()
                            
                            for make in common_makes:
                                make_upper = make.upper()
                                # Check if make appears in the string
                                idx = pdf_car_upper.find(make_upper)
                                if idx != -1:
                                    # Found a make - use it
                                    found_make = make
                                    make_start_idx = idx
                                    break
                            
                            if found_make and make_start_idx is not None:
                                # Extract everything from the make onwards as the car description
                                car_desc = pdf_car[make_start_idx:].strip()
                                # Remove the make name from the start to get the model
                                # Handle multi-word makes like "Aston Martin"
                                make_words = found_make.split()
                                car_desc_words = car_desc.split()
                                
                                # Check if the first word(s) match the make
                                if len(car_desc_words) >= len(make_words):
                                    # Check if first words match make
                                    matches = all(car_desc_words[i].upper() == make_words[i].upper() 
                                                 for i in range(len(make_words)))
                                    if matches:
                                        # Make matches - rest is model
                                        model_parts = car_desc_words[len(make_words):]
                                        parsed['Make'] = found_make
                                        parsed['Model'] = ' '.join(model_parts) if model_parts else ''
                                    else:
                                        # First word doesn't match - try removing make string
                                        parsed['Make'] = found_make
                                        model_str = car_desc.replace(found_make, '', 1).strip()
                                        parsed['Model'] = model_str if model_str else car_desc
                                else:
                                    # Not enough words - use make and try to extract model
                                    parsed['Make'] = found_make
                                    model_str = car_desc.replace(found_make, '', 1).strip()
                                    parsed['Model'] = model_str if model_str else ''
                            else:
                                # No known make found - try simple split on first space
                                car_parts = pdf_car.split(' ', 1)
                                if len(car_parts) >= 2:
                                    parsed['Make'] = car_parts[0]
                                    parsed['Model'] = car_parts[1]
                                else:
                                    # Can't parse - use entire string as Model
                                    parsed['Model'] = pdf_car
                            
                            parsed['Car'] = pdf_car  # Keep full car string in 'Car' field
                            _log(f"[RACE-TAG] Overriding AI Make/Model with PDF data: Make='{parsed.get('Make')}', Model='{parsed.get('Model')}', Full='{pdf_car}'")
                        if matched_data.get('team'):
                            parsed['Team'] = matched_data['team']
                        if matched_data.get('driver'):
                            parsed['Driver'] = matched_data['driver']
                        if matched_data.get('session'):
                            parsed['Session'] = matched_data['session']
                    else:
                        _log(f"[RACE-TAG] Race number '{race_number}' not found in PDF data. Available numbers: {list(self.race_data.keys())[:10]}...")
                else:
                    _log(f"[RACE-TAG] No valid race number extracted (got '{race_number}'), skipping PDF match")
                
                # Match image timestamp to session (if no session from race number match)
                # This identifies which heat the image was taken during
                if 'Session' not in parsed or not parsed.get('Session'):
                    matched_session = match_image_to_session(image_path, self.race_sessions)
                    if matched_session:
                        parsed['Session'] = matched_session
                        _log(f"[RACE-TAG] Matched image timestamp to session: {matched_session}")
                    else:
                        _log(f"[RACE-TAG] Could not match image timestamp to any session")
                
                # Parse Session to extract Event and Heat
                # This works for both race number matches and timestamp matches
                session_str = parsed.get('Session', '')
                if session_str:
                    import re
                    # Pattern: "Event/Class - Heat X" or "Event/Class - Heat Xb" (e.g., "Clubman, Modified 4x4 & BMW Mini - Heat 1")
                    # Also handle formats like "Supermodified - Heat 1" or "MSUK & National Cross Cars - Heat 1b"
                    heat_match = re.search(r'-\s*Heat\s+(\d+[a-z]?)', session_str, re.IGNORECASE)
                    if heat_match:
                        parsed['Heat'] = heat_match.group(1)
                        # Extract event/class (everything before " - Heat")
                        event_match = re.search(r'^(.+?)\s*-\s*Heat', session_str, re.IGNORECASE)
                        if event_match:
                            parsed['Event'] = event_match.group(1).strip()
                        else:
                            # If no " - Heat" pattern, try to extract from other patterns
                            # Fallback: use session as event if no clear heat pattern
                            parsed['Event'] = session_str
                        _log(f"[RACE-TAG] Extracted Event='{parsed.get('Event')}' and Heat='{parsed.get('Heat')}' from session '{session_str}'")
                    else:
                        # No heat found, use entire session as event
                        parsed['Event'] = session_str
                        _log(f"[RACE-TAG] No heat pattern found in session '{session_str}', using as Event only")
                
                # If we still don't have Heat but have a timestamp, try to match to any session
                # This ensures we identify the heat even if race number matching failed
                if 'Heat' not in parsed or not parsed.get('Heat'):
                    img_datetime = read_image_datetime(image_path)
                    if img_datetime:
                        # Find the session that contains this timestamp
                        for session_name, (start_dt, end_dt) in self.race_sessions.items():
                            if start_dt <= img_datetime <= end_dt:
                                # Extract heat from this session name
                                import re
                                heat_match = re.search(r'-\s*Heat\s+(\d+[a-z]?)', session_name, re.IGNORECASE)
                                if heat_match:
                                    parsed['Heat'] = heat_match.group(1)
                                    if 'Session' not in parsed or not parsed.get('Session'):
                                        parsed['Session'] = session_name
                                    if 'Event' not in parsed or not parsed.get('Event'):
                                        event_match = re.search(r'^(.+?)\s*-\s*Heat', session_name, re.IGNORECASE)
                                        if event_match:
                                            parsed['Event'] = event_match.group(1).strip()
                                    _log(f"[RACE-TAG] Identified Heat='{parsed.get('Heat')}' from timestamp match to session '{session_name}'")
                                break
            except Exception as e:
                _log(f"[RACE-TAG] Error applying race metadata: {e}")
                import traceback
                _log(traceback.format_exc())
        
        # If race metadata was added, write metadata again to include Team/Driver/Session/Event/Heat/Track/EventName
        if self.race_tagging_enabled and self.race_data and self.race_sessions:
            if any(key in parsed for key in ['Team', 'Driver', 'Session', 'Car', 'Event', 'Heat', 'Race Track', 'Event Name']):
                try:
                    _log(f"[RACE-TAG] Re-writing metadata with race data: Team={parsed.get('Team')}, Driver={parsed.get('Driver')}, Session={parsed.get('Session')}, Event={parsed.get('Event')}, Heat={parsed.get('Heat')}, Track={parsed.get('Race Track')}, EventName={parsed.get('Event Name')}")
                    write_metadata_to_image(image_path, parsed)
                except Exception as e:
                    _log(f"[RACE-TAG] Error re-writing metadata: {e}")
        
        # Check if this is a retry result and splash is open - update splash instead of adding to list
        if self._current_splash and image_path in self._current_splash.rejected_paths:
            # This is a retry result - add it to splash with Accept button
            self._current_splash.add_retry_result(image_path, parsed, raw)
            # Don't add to results_list yet - wait for acceptance
        else:
            # Normal result - add to results list
            item = QtWidgets.QListWidgetItem()
            item_widget = ResultItemWidget(image_path, parsed, raw)
            try:
                item_widget.rejected.connect(self.on_result_rejected)
            except Exception:
                pass
            item.setSizeHint(item_widget.sizeHint())
            # Add to bottom (not top) to prevent UI redrawing/jumping
            # Use insertItem with count() to explicitly add at the end
            insert_pos = self.results_list.count()
            self.results_list.insertItem(insert_pos, item)
            self.results_list.setItemWidget(item, item_widget)
            # Auto-scroll to bottom to show newly added item
            self.results_list.scrollToBottom()
            # Keep all history - no item removal
            
            # Store result for batch splash screen
            if self.is_batch:
                self._batch_results.append({
                    'image_path': image_path,
                    'parsed': parsed,
                    'raw': raw
                })
        
        if self.is_batch:
            self.batch_done += 1
            self.progress.setVisible(True)
            self.progress.setRange(0, self.batch_total)
            self.progress.setValue(self.batch_done + self.batch_failed + self.batch_skipped)
            self.status.showMessage(f'Completed: {os.path.basename(image_path)}  ({self.batch_done + self.batch_failed}/{self.batch_total})', 4000)
            # Refill pending window from remaining files
            self._refill_pending_window()
            self._check_batch_completion_and_finish()
        else:
            self.status.showMessage(f'Completed: {os.path.basename(image_path)}', 4000)
        self._console_log(f"[DONE]  {image_path}")
            # hide progress if no more inflight
        if not self.inflight and not self.is_batch:
                self.progress.setVisible(False)
                self.progress.setRange(0, 1)
                try:
                    self._set_activity(False)
                except Exception:
                    pass
        # Update remaining counter
        try:
            self._update_remaining_label()
        except Exception:
            pass
        # Concurrency ramp up on success (but never exceed user's setting)
        self.success_streak += 1
        self.failure_streak = 0
        if self.success_streak >= 3 and self.dynamic_limit < self.max_concurrent:
            self.dynamic_limit = min(self.dynamic_limit + 1, self.max_concurrent)
            self.success_streak = 0
        # Try to drain any queued work if capacity available
        self._drain_queue()

    @Slot(dict)
    def on_result_rejected(self, data: dict):
        """Receive per-item rejection and immediately re-submit with feedback. Handle KB building if requested."""
        try:
            image_path = str(data.get('image_path') or '')
            if not image_path:
                return
            
            # Handle KB building if requested
            add_to_kb = data.get('add_to_kb', False)
            correct_make = data.get('correct_make', '').strip()
            correct_model = data.get('correct_model', '').strip()
            
            if add_to_kb and correct_make and correct_model:
                try:
                    # Extract distinguishing features using AI
                    features = extract_distinguishing_features_ai(
                        self.client, self.model_name, image_path, correct_make, correct_model
                    )
                    
                    # Save reference image
                    ref_image_path = save_reference_image(image_path, correct_make, correct_model)
                    
                    # Save to user KB
                    user_kb_path = os.path.join(_app_base_dir(), 'user_knowledge_base.json')
                    save_to_user_knowledge_base(user_kb_path, correct_make, correct_model, features, ref_image_path)
                    
                    _log(f"[KB-BUILDER] Added {correct_make} {correct_model} to user KB")
                except Exception as e:
                    _log(f"[KB-BUILDER] Error adding to KB: {e}")
                    import traceback
                    _log(traceback.format_exc())
            
            # Build feedback context - ALWAYS include correct_make/model if provided (even if not adding to KB)
            feedback_context = {
                'reason': data.get('reason',''), 
                'comments': data.get('comments','')
            }
            
            # If user provided correct make/model, ALWAYS include it in feedback (whether or not adding to KB)
            if correct_make and correct_model:
                feedback_context['correct_make'] = correct_make
                feedback_context['correct_model'] = correct_model
                
                # If KB entry was added, also include distinguishing cues
                if add_to_kb:
                    try:
                        user_kb = load_user_knowledge_base()
                        if correct_make in user_kb:
                            for entry in user_kb[correct_make]:
                                if isinstance(entry, dict) and entry.get('model', '').lower() == correct_model.lower():
                                    cues = entry.get('distinguishing_cues', {})
                                    if cues:
                                        feedback_context['distinguishing_cues'] = cues
                                    break
                    except Exception:
                        pass
                else:
                    # Even if not adding to KB, try to get cues from existing KB entry
                    try:
                        user_kb = load_user_knowledge_base()
                        approved_kb = load_approved_knowledge_base()
                        builtin_kb = load_vehicle_knowledge_base(self.vehicle_kb_path) if self.vehicle_kb_path else {}
                        merged_kb = merge_knowledge_bases(builtin_kb, approved_kb, user_kb)
                        
                        if correct_make in merged_kb:
                            for entry in merged_kb[correct_make]:
                                if isinstance(entry, dict) and entry.get('model', '').lower() == correct_model.lower():
                                    cues = entry.get('distinguishing_cues', {})
                                    if cues:
                                        feedback_context['distinguishing_cues'] = cues
                                    break
                    except Exception:
                        pass
            
            # Immediately re-submit this image with feedback context
            task = InferenceTask(self.client, self.model_name, image_path, self.embed_checkbox.isChecked())
            task.feedback_context = feedback_context
            try:
                task.motorshow_mode = bool(self.motorshow_checkbox.isChecked())
            except Exception:
                task.motorshow_mode = False
            # Auto-enable race mode if race tagging is enabled and PDF is loaded
            task.race_mode = bool(self.race_tagging_enabled and self.race_data)
            worker = TaskRunnable(task)
            task.finished.connect(self.on_task_finished)
            task.failed.connect(self.on_task_failed)
            self.inflight.add(image_path)
            self._active_tasks[image_path] = (task, worker)
            self._console_log(f"[RETRY-FEEDBACK] {image_path} reason={data.get('reason','')} comments_len={len(str(data.get('comments','')))} kb={add_to_kb}")
            self.threadpool.start(worker)
            self.status.showMessage(f"Re-submitting with feedback: {os.path.basename(image_path)}", 4000)
            # Reflect as processing now (assign slot, start timing, show progress if single mode)
            try:
                self._assign_processing_slot(image_path)
                self.task_started_at[image_path] = time.time()
                if not self.is_batch:
                    self.progress.setVisible(True)
                    self.progress.setRange(0, 0)
            except Exception:
                pass
        except Exception as e:
            _log(f"Error in on_result_rejected: {e}")
            import traceback
            _log(traceback.format_exc())
    
    @Slot(dict)
    def on_result_accepted(self, data: dict):
        """Receive acceptance of a retry result - add to results list."""
        try:
            image_path = data.get('image_path', '')
            parsed = data.get('parsed', {})
            raw = data.get('raw', '')
            
            if not image_path:
                return
            
            # Add to results list
            item = QtWidgets.QListWidgetItem()
            item_widget = ResultItemWidget(image_path, parsed, raw)
            try:
                item_widget.rejected.connect(self.on_result_rejected)
            except Exception:
                pass
            item.setSizeHint(item_widget.sizeHint())
            insert_pos = self.results_list.count()
            self.results_list.insertItem(insert_pos, item)
            self.results_list.setItemWidget(item, item_widget)
            self.results_list.scrollToBottom()
            
            # Keep last 20
            if self.results_list.count() > 20:
                self.results_list.takeItem(0)
        except Exception as e:
            _log(f"Error in on_result_accepted: {e}")
            import traceback
            _log(traceback.format_exc())

    @Slot(str, str)
    def on_task_failed(self, image_path: str, error: str):
        self.inflight.discard(image_path)
        self._active_tasks.pop(image_path, None)
        self._clear_processing_slot(image_path)
        # Update rolling timing if we had a start time
        started = self.task_started_at.pop(image_path, None)
        if isinstance(started, (int, float)):
            self._record_duration(time.time() - started)
        if self.is_batch:
            self.batch_failed += 1
            self.progress.setVisible(True)
            self.progress.setRange(0, self.batch_total)
            self.progress.setValue(self.batch_done + self.batch_failed + self.batch_skipped)
            self.status.showMessage(f'Error: {os.path.basename(image_path)} - {error}  ({self.batch_done + self.batch_failed}/{self.batch_total})', 8000)
            # Remember failed path for potential retry at end
            try:
                self._failed_paths.append(image_path)
            except Exception:
                pass
            # Refill pending window from remaining files
            self._refill_pending_window()
            self._check_batch_completion_and_finish()
        else:
            self.status.showMessage(f'Error: {os.path.basename(image_path)} - {error}', 8000)
        self._console_log(f"[FAIL] {image_path} : {error}")
        if not self.inflight and not self.is_batch:
                self.progress.setVisible(False)
                self.progress.setRange(0, 1)
                try:
                    self._set_activity(False)
                except Exception:
                    pass
        # Concurrency backoff on failure
        self.failure_streak += 1
        self.success_streak = 0
        if self.failure_streak >= 2 and self.dynamic_limit > 1:
            self.dynamic_limit -= 1
            self.failure_streak = 0
        # Try to drain any queued work if capacity available
        self._drain_queue()
        # Update remaining counter
        try:
            self._update_remaining_label()
        except Exception:
            pass

    def batch_process_folder(self, folder: str):
        # Support all common image formats that exiftool can handle
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp', '.heic', '.heif',
                '.cr2', '.nef', '.arw', '.raf', '.orf', '.rw2', '.dng', '.raw', '.tiff'}
        # Initialize batch tracking and progress (streaming discovery)
        self.is_batch = True
        self._batch_results = []  # Clear batch results for new batch
        self.batch_total = 0  # unknown upfront; will grow as we stream
        self.batch_done = 0
        self.batch_failed = 0
        self.batch_skipped = 0
        self.dynamic_limit = self.max_concurrent  # Use user's setting directly
        # Ensure paused state is reset when starting new batch
        self.paused = False
        self.btn_pause.setText('Pause')
        self.success_streak = 0
        self.failure_streak = 0
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate until we count
        self.progress.setValue(0)
        self.status.showMessage('Starting batch (streaming)...')
        self._console_log(f"[BATCH] streaming walk window={self.batch_window_size} concurrency={self.max_concurrent}")
        # Reset queues and create directory iterator
        self._pending_queue.clear()
        self._batch_remaining = []
        def _iter_files():
            base = Path(folder)
            if self.recursive_checkbox.isChecked():
                for root, _dirs, files_local in os.walk(base):
                    for name in files_local:
                        if Path(name).suffix.lower() in exts:
                            yield str(Path(root) / name)
            else:
                for p in base.glob('*'):
                    if p.suffix.lower() in exts:
                        yield str(p)
        self._dir_iter = _iter_files()
        # Stage first window and drain
        self._refill_pending_window()
        self._drain_queue()
        # Handle empty folder gracefully with a dialog
        try:
            if self.batch_total == 0 and not self._pending_queue and not self.inflight:
                self.is_batch = False
                self.progress.setVisible(False)
                self.progress.setRange(0, 1)
                QtWidgets.QMessageBox.information(self, 'Batch Complete', 'No images found in selected folder.')
        except Exception:
            pass

    # Processing slots helpers
    def _assign_processing_slot(self, path: str):
        for lbl in self.processing_slots:
            if not lbl.isVisible():
                continue
            if lbl.property('busy') != True:
                try:
                    # Load thumbnail
                    img = Image.open(path)
                    w, h = img.size
                    # Fit to available label size while preserving aspect ratio
                    tgt_w = max(1, lbl.width() or lbl.minimumWidth() or 220)
                    tgt_h = max(1, lbl.height() or lbl.minimumHeight() or 165)
                    scale = min(tgt_w / max(1, w), tgt_h / max(1, h), 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.BOX)
                    buf = io.BytesIO(); img.save(buf, format='PNG')
                    qimg = QtGui.QImage.fromData(buf.getvalue(), 'PNG')
                    pix = QtGui.QPixmap.fromImage(qimg)
                    # If label grows later, let Qt upscale smoothly
                    lbl.setScaledContents(False)
                    lbl.setPixmap(pix)
                except Exception:
                    lbl.setText(os.path.basename(path))
                lbl.setProperty('busy', True)
                lbl.setToolTip(os.path.basename(path))
                return

    def _clear_processing_slot(self, path: str):
        for lbl in self.processing_slots:
            if lbl.property('busy') == True and lbl.toolTip() == os.path.basename(path):
                lbl.setPixmap(QtGui.QPixmap())
                lbl.setText('Idle')
                lbl.setProperty('busy', False)
                lbl.setToolTip('')
                return

    def _update_processing_slots_visibility(self):
        try:
            # Show 1 or 2 slots when max_concurrent <= 2, otherwise show 4
            desired = 4 if int(self.max_concurrent) > 2 else max(1, int(self.max_concurrent))
            for idx, lbl in enumerate(self.processing_slots):
                lbl.setVisible(idx < desired)
        except Exception:
            pass

    @Slot(int)
    def on_workers_changed(self, value: int):
        try:
            self.max_concurrent = int(value)
            # Apply immediately rather than waiting for ramp-up
            self.dynamic_limit = int(value)
            # Update thread pool max thread count (add headroom for metadata checks)
            self.threadpool.setMaxThreadCount(max(int(value) + 2, 4))
            self._update_processing_slots_visibility()
            # Try to start more work right away
            self._drain_queue()
            # Persist user preference for next launch
            try:
                s = QSettings('Pistonspy', 'TagOmatic')
                s.setValue('ui/max_concurrent', int(value))
            except Exception:
                pass
        except Exception:
            pass
    @Slot(str)
    def on_workers_combo_changed(self, text: str):
        try:
            if not text:
                return
            self.on_workers_changed(int(text))
        except Exception:
            pass
    # Metadata existence check (best-effort)
    def _metadata_exists(self, image_path: str, fast_only: bool = False) -> bool:
        try:
            # Support all image formats that exiftool can read (JPEG, TIFF, PNG, RAW, HEIC, etc.)
            # Fast signature check: look for our sentinel keys in the binary header to avoid exiftool on UNC
            try:
                with open(image_path, 'rb') as f:
                    buf = f.read(512*1024)
                hay = buf.decode('latin-1', errors='ignore')
                if ('AI-Interpretation Summary' in hay) or ('LLM Backend' in hay):
                    return True
            except Exception:
                pass
            # Fast path: PIL EXIF only
            try:
                img = Image.open(image_path)
                exif = img.getexif()
                if 0x9286 in exif and str(exif[0x9286]).strip():
                    return True
            except Exception:
                pass
            if fast_only:
                # Avoid spawning exiftool during large batch scheduling on UI thread
                return False
            # Full check with exiftool JSON (blocking; only used outside batch)
            try:
                import subprocess, json as _json
                exe = _find_exiftool()
                if exe:
                    cmd = [exe, '-EXIF:UserComment', '-IPTC:Keywords', '-IPTC:ObjectName', '-IPTC:Caption-Abstract', '-j', str(image_path)]
                    kwargs = _exiftool_run_kwargs(exe, timeout=20)
                    res = subprocess.run(cmd, **kwargs)
                    if res.returncode == 0 and res.stdout.strip():
                        data = _json.loads(res.stdout)
                        if isinstance(data, list) and data and ('EXIF:UserComment' in data[0] or 'UserComment' in data[0]):
                            val = data[0].get('EXIF:UserComment') or data[0].get('UserComment')
                            if val and str(val).strip():
                                return True
                        # Consider IPTC we control as a sign of existing metadata
                        kw = data[0].get('IPTC:Keywords') if isinstance(data[0], dict) else None
                        objn = data[0].get('IPTC:ObjectName') if isinstance(data[0], dict) else None
                        cap = data[0].get('IPTC:Caption-Abstract') if isinstance(data[0], dict) else None
                        if (kw and str(kw).strip()) or (objn and str(objn).strip()) or (cap and str(cap).strip()):
                            return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    # Timeout watchdog
    def _check_timeouts(self):
        now = time.time()
        for path in list(self.inflight):
            started = self.task_started_at.get(path)
            if started and (now - started) > self.per_image_timeout_secs:
                # Mark timed out
                self.timed_out.add(path)
                self.inflight.discard(path)
                self._active_tasks.pop(path, None)
                self.batch_failed += 1 if self.is_batch else 0
                self._clear_processing_slot(path)
                self.status.showMessage(f'Timeout: {os.path.basename(path)}', 6000)
                # Backoff concurrency
                self.failure_streak += 1
                if self.dynamic_limit > 1:
                    self.dynamic_limit -= 1
                # Retry once if not exceeded
                retries = getattr(self, '_retries', {})
                count = retries.get(path, 0)
                if count < 1:
                    retries[path] = count + 1
                    setattr(self, '_retries', retries)
                    QtCore.QTimer.singleShot(1000, lambda p=path: self.process_image(p))
                else:
                    # Finalize batch progress bar and record failure for retry option
                    if self.is_batch:
                        try:
                            self._failed_paths.append(path)
                        except Exception:
                            pass
                        self.progress.setValue(self.batch_done + self.batch_failed)
                        self._check_batch_completion_and_finish()

    def _check_batch_completion_and_finish(self):
        # Consider skipped items towards completion as they represent handled files
        completed_count = self.batch_done + self.batch_failed + self.batch_skipped
        if self.is_batch and (completed_count) >= self.batch_total and not self.inflight:
            self.is_batch = False
            # Reset paused state when batch completes
            self.paused = False
            self.btn_pause.setText('Pause')
            # Offer a retry for failed items once
            if self._failed_paths and self._retry_round == 0:
                try:
                    count = len(self._failed_paths)
                    res = QtWidgets.QMessageBox.question(
                        self,
                        'Batch Complete',
                        f'Batch complete: {self.batch_done}/{self.batch_total} processed, {self.batch_failed} failed.\n\nRetry {count} failed item(s) now?',
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes,
                    )
                    if res == QtWidgets.QMessageBox.Yes:
                        self._start_retry_failed_batch()
                        return
                except Exception:
                    pass
            summary = f'Batch complete: {self.batch_done}/{self.batch_total} processed'
            if self.batch_skipped:
                summary += f', {self.batch_skipped} skipped'
            if self.batch_failed:
                summary += f', {self.batch_failed} failed'
            self.status.showMessage(summary, 8000)
            self.progress.setVisible(False)
            self.progress.setRange(0, 1)
            
            # Show splash screen with batch results
            if self._batch_results:
                try:
                    splash = BatchResultsSplashDialog(self, self._batch_results)
                    splash.rejected.connect(self.on_result_rejected)
                    splash.accepted.connect(self.on_result_accepted)
                    self._current_splash = splash
                    splash.exec()
                    self._current_splash = None
                except Exception as e:
                    _log(f"Error showing batch splash screen: {e}")
                    import traceback
                    _log(traceback.format_exc())
                    QtWidgets.QMessageBox.information(self, 'Batch Complete', summary)
            else:
                QtWidgets.QMessageBox.information(self, 'Batch Complete', summary)
            
            self._retry_round = 0
            # Final update to remaining label (should read 0)
            try:
                self._update_remaining_label()
            except Exception:
                pass

    def _start_retry_failed_batch(self):
        """Initialize a new mini-batch over the previously failed paths."""
        try:
            to_retry: list[str] = []
            seen = set()
            for p in self._failed_paths:
                if p not in seen:
                    seen.add(p)
                    to_retry.append(p)
            self._failed_paths = []
            if not to_retry:
                return
            self._retry_round = 1
            # Reset batch state for retry
            self.is_batch = True
            self.batch_total = len(to_retry)
            self.batch_done = 0
            self.batch_failed = 0
            self.batch_skipped = 0
            self.dynamic_limit = self.max_concurrent  # Use user's setting directly
            # Ensure paused state is reset when starting retry batch
            self.paused = False
            self.btn_pause.setText('Pause')
            self.success_streak = 0
            self.failure_streak = 0
            self._pending_queue.clear()
            self._batch_remaining = []
            self._dir_iter = None
            # Clear timeout/timed_out markers and retry counters for these files
            try:
                retries = getattr(self, '_retries', {})
            except Exception:
                retries = {}
            for p in to_retry:
                self.timed_out.discard(p)
                try:
                    if p in retries:
                        del retries[p]
                except Exception:
                    pass
                self._pending_queue.append(p)
            setattr(self, '_retries', retries)
            # Show progress for retry batch
            self.progress.setVisible(True)
            self.progress.setRange(0, self.batch_total)
            self.progress.setValue(0)
            self.status.showMessage(f'Retrying {len(to_retry)} failed item(s)...')
            self._console_log(f"[BATCH-RETRY] {len(to_retry)} items")
            # Start processing
            self._drain_queue()
        except Exception:
            pass

    # Pause/Resume and Emergency Stop
    def _drain_queue(self):
        try:
            if self.paused:
                return
            effective_limit = min(self.max_concurrent, self.dynamic_limit)
            while self._pending_queue and len(self.inflight) < effective_limit and not self.paused:
                nxt = self._pending_queue.pop(0)
                # process_image will honour pause if toggled mid-drain
                self.process_image(nxt)
        except Exception:
            pass

    def _apply_blue_theme(self):
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        app.setStyleSheet('''
            QMainWindow, QWidget { background-color: #0b1220; color: #dbeafe; }
            QLabel, QCheckBox, QRadioButton { color: #dbeafe; }
            QGroupBox { border: 1px solid #1e3a8a; border-radius: 4px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #93c5fd; }
            QPushButton { background-color: #1e3a8a; color: #ffffff; border: 1px solid #2563eb; border-radius: 4px; padding: 6px 10px; }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #2a2e35; color: #8aa0c8; border-color: #2a2e35; }
            QComboBox, QSpinBox, QLineEdit, QPlainTextEdit, QListWidget { background-color: #0f172a; color: #e2e8f0; border: 1px solid #1f2937; border-radius: 4px; }
            QStatusBar { 
                background-color: #0b0f14; 
                color: #dbeafe; 
                border: none; 
                outline: none;
            }
            QStatusBar::item { 
                border: none; 
                outline: none;
            }
            QLabel { 
                border: none; 
                outline: none;
            }
            QProgressBar { 
                background-color: #0f172a; 
                border: none; 
                border-radius: 8px; 
                text-align: center; 
                color: #dbeafe; 
                padding: 0px;
                outline: none;
                margin: 0px;
                min-height: 20px;
                max-height: 20px;
                selection-background-color: transparent;
            }
            QProgressBar::chunk { 
                background-color: #2563eb; 
                border-radius: 8px; 
                margin: 0px;
                border: none;
                outline: none;
                padding: 0px;
                min-height: 20px;
                max-height: 20px;
                selection-background-color: transparent;
            }
        ''')

    def _refill_pending_window(self):
        try:
            if not self.is_batch:
                return
            # Keep at most batch_window_size tasks across inflight + pending
            target = max(0, self.batch_window_size - (len(self.inflight) + len(self._pending_queue) + len(self._active_meta)))
            moved = 0
            # Pull from preloaded remainder first
            while target > 0 and self._batch_remaining:
                nxt = self._batch_remaining.pop(0)
                if nxt not in self._pending_queue and nxt not in self.inflight:
                    self._pending_queue.append(nxt)
                    target -= 1
                    moved += 1
            # Then stream from iterator
            if target > 0 and self._dir_iter is not None:
                try:
                    while target > 0:
                        nxt = next(self._dir_iter)
                        self.batch_total += 1
                        if nxt not in self._pending_queue and nxt not in self.inflight:
                            self._pending_queue.append(nxt)
                            target -= 1
                            moved += 1
                except StopIteration:
                    self._dir_iter = None
                except Exception:
                    pass
            if moved:
                if self.batch_total > 0:
                    self.progress.setRange(0, self.batch_total)
                self.status.showMessage(f'Queued window: {len(self.inflight) + len(self._pending_queue)}/{self.batch_total or "?"}')
                self._drain_queue()
                # Update remaining counter
                try:
                    self._update_remaining_label()
                except Exception:
                    pass
        except Exception:
            pass

    def _record_duration(self, seconds: float):
        try:
            self.recent_durations.append(float(seconds))
            if len(self.recent_durations) > 3:
                self.recent_durations.pop(0)
            avg = sum(self.recent_durations) / max(1, len(self.recent_durations))
            self.avg_label.setText(f'Avg last 3: {avg:.2f}s')
        except Exception:
            pass

    def _update_remaining_label(self):
        """Compute and display remaining items in batch (pending + inflight + active meta)."""
        try:
            if not self.is_batch:
                self.remaining_label.setText('Remaining: --')
                return
            # Queue-derived view (most trustworthy at end of batch)
            try:
                q_remaining = (
                    len(self._batch_remaining) +
                    len(self._pending_queue) +
                    len(self.inflight) +
                    len(self._active_meta)
                )
            except Exception:
                q_remaining = 0
            # If nothing is pending or inflight, clamp to 0 regardless of stale totals
            if q_remaining == 0:
                self.remaining_label.setText('Remaining: 0')
                return
            # Total-derived view (good early in batch when total is known)
            t_remaining = 0
            try:
                t_remaining = max(0, int(self.batch_total) - int(self.batch_done + self.batch_failed + self.batch_skipped))
            except Exception:
                t_remaining = q_remaining
            # Show the higher of the two to avoid under-reporting mid-batch
            remaining = max(q_remaining, t_remaining)
            self.remaining_label.setText(f'Remaining: {remaining}')
        except Exception:
            try:
                self.remaining_label.setText('Remaining: --')
            except Exception:
                pass

    @Slot()
    def on_pause_clicked(self):
        try:
            self.paused = not self.paused
            if self.paused:
                self.btn_pause.setText('Resume')
                self.status.showMessage('Paused scheduling', 4000)
            else:
                self.btn_pause.setText('Pause')
                self.status.showMessage('Resumed scheduling', 3000)
                # Drain queued items now that we resumed
                self._refill_pending_window()
                QtCore.QTimer.singleShot(0, self._drain_queue)
        except Exception:
            pass

    @Slot()
    def on_emergency_stop_clicked(self):
        try:
            # Enter paused state and clear pending queue
            self.paused = True
            self.btn_pause.setText('Resume')
            self._pending_queue.clear()
            self._batch_remaining.clear()
            # Cancel/forget any metadata checks
            self._active_meta.clear()
            # Mark inflight as timed out so completions are ignored
            for path in list(self.inflight):
                self.timed_out.add(path)
                self._clear_processing_slot(path)
            self.inflight.clear()
            self._active_tasks.clear()
            # Reset batch progress if running
            if self.is_batch:
                self.progress.setVisible(False)
                self.progress.setRange(0, 1)
                self.is_batch = False
            self.status.showMessage('Emergency stop: cancelled queued work and ignoring inflight tasks', 6000)
        except Exception:
            pass

    @Slot()
    def on_reject_clicked(self):
        """Handle reject button click with enhanced feedback dialog"""
        try:
            if not hasattr(self, 'current_result') or not self.current_result:
                QtWidgets.QMessageBox.information(self, 'No Result', 'No result available to reject.')
                return
            # Create reject dialog
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Reject Result - Provide Feedback')
            dlg.resize(500, 400)

            layout = QtWidgets.QVBoxLayout(dlg)

            # Feedback options
            feedback_group = QtWidgets.QGroupBox('Rejection Reason')
            feedback_layout = QtWidgets.QVBoxLayout(feedback_group)

            reasons = [
                'Incorrect make/model',
                'Poor image quality',
                'No car visible',
                'Wrong vehicle type',
                'Incomplete result',
                'Other (specify below)'
            ]

            reason_buttons = []
            for reason in reasons:
                btn = QtWidgets.QRadioButton(reason)
                reason_buttons.append(btn)
                feedback_layout.addWidget(btn)

            # Select first reason by default
            if reason_buttons:
                reason_buttons[0].setChecked(True)

            layout.addWidget(feedback_group)

            # Additional comments
            comment_label = QtWidgets.QLabel('Additional Comments:')
            comment_text = QtWidgets.QTextEdit()
            comment_text.setMaximumHeight(100)
            layout.addWidget(comment_label)
            layout.addWidget(comment_text)

            # Buttons
            btn_layout = QtWidgets.QHBoxLayout()
            reject_btn = QtWidgets.QPushButton('Confirm Rejection')
            reject_btn.setStyleSheet('QPushButton { background-color: #dc2626; color: #fff; }')
            cancel_btn = QtWidgets.QPushButton('Cancel')

            reject_btn.clicked.connect(dlg.accept)
            cancel_btn.clicked.connect(dlg.reject)

            btn_layout.addWidget(reject_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addLayout(btn_layout)

            # Show dialog
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                # Get selected reason
                selected_reason = 'Unknown'
                for i, btn in enumerate(reason_buttons):
                    if btn.isChecked():
                        selected_reason = reasons[i]
                        break

                # Get comments
                comments = comment_text.toPlainText().strip()

                # Log rejection for improvement
                rejection_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'image_path': getattr(self, 'image_path', 'Unknown'),
                    'result': self.current_result,
                    'reason': selected_reason,
                    'comments': comments
                }

                # Save rejection log
                try:
                    with open('rejection_log.json', 'a') as f:
                        json.dump(rejection_data, f)
                        f.write('\n')
                except Exception as e:
                    print(f"Error saving rejection log: {e}")

                # Show confirmation
                QtWidgets.QMessageBox.information(self, 'Result Rejected',
                    f'Result rejected successfully.\nReason: {selected_reason}\nComments: {comments or "None"}')

                # Clear current result
                self.current_result = None
                self.reject_btn.setEnabled(False)
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application icon for taskbar (CRITICAL for Windows taskbar display)
    try:
        app_icon_path = _resource_path('tagomatic_main.ico')
        app_icon = QtGui.QIcon(app_icon_path)
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
            print(f"[INFO] Application icon set: {app_icon_path}")
        else:
            print(f"[WARNING] Could not load application icon: {app_icon_path}")
    except Exception as e:
        print(f"[ERROR] Failed to set application icon: {e}")
    
    # Optional splash screen with logo
    splash = None
    try:
        # Use dedicated startup splash logo (JPEG)
        logo_path = _resource_path('logo.jpg')
        pix = QtGui.QPixmap(logo_path)
        if not pix.isNull():
            scaled = pix.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            splash = QtWidgets.QSplashScreen(scaled)
            splash.show()
            app.processEvents()
    except Exception:
        splash = None

    w = MainWindow()
    # Close splash shortly after window is ready
    if splash is not None:
        QtCore.QTimer.singleShot(800, splash.close)
        # Delay version check until after splash closes
        QtCore.QTimer.singleShot(1000, w._check_latest_version_async)
    else:
        # No splash, check version immediately
        QtCore.QTimer.singleShot(2000, w._check_latest_version_async)
    w.show()
    sys.exit(app.exec())

# --- Background task classes ---
# Define these classes before main() so they're available for type hints
if True:  # Always define these classes
    class InferenceTask(QtCore.QObject):
        finished = Signal(str, dict, str)
        failed = Signal(str, str)
        def __init__(self, client, model_name, image_path, embed):
            super().__init__()
            self.client = client
            self.model_name = model_name
            self.image_path = image_path
            self.embed = embed
            self.feedback_context = None
            self.motorshow_mode = False
            self.motorshow_disabled = False
            self.race_mode = False
            self.noncar_model = None
        @QtCore.Slot()
        def run(self):
            try:
                raw, parsed = infer_enhanced(
                    self.client, self.model_name, self.image_path,
                    feedback_context=self.feedback_context,
                    motorshow_mode=self.motorshow_mode,
                    motorshow_disabled=self.motorshow_disabled,
                    race_mode=self.race_mode,
                    noncar_model=self.noncar_model,
                )
                if not isinstance(raw, str) and not isinstance(parsed, dict):
                    # If infer_enhanced returned a single object or None, coerce to safe defaults
                    try:
                        tmp = infer_enhanced(
                            self.client, self.model_name, self.image_path,
                            feedback_context=self.feedback_context,
                            motorshow_mode=self.motorshow_mode,
                            motorshow_disabled=self.motorshow_disabled,
                            race_mode=self.race_mode,
                            noncar_model=self.noncar_model,
                        )
                        if isinstance(tmp, tuple) and len(tmp) == 2:
                            raw, parsed = tmp
                        else:
                            raw, parsed = (str(tmp) if tmp is not None else ''), {}
                    except Exception:
                        raw, parsed = '', {}

                if self.embed:
                    try:
                        write_metadata_to_image(self.image_path, parsed)
                    except Exception:
                        pass
                self.finished.emit(self.image_path, parsed, raw)
            except Exception as e:
                self.failed.emit(self.image_path, str(e))
    class TaskRunnable(QtCore.QRunnable):
        def __init__(self, task):
            super().__init__()
            self.task = task
            self.setAutoDelete(True)
        def run(self):
            self.task.run()
    class MetaCheckTask(QtCore.QObject):
        finished = Signal(str, bool)
        def __init__(self, image_path):
            super().__init__()
            self.image_path = image_path
        @QtCore.Slot()
        def run(self):
            exists = False
            try:
                try:
                    with open(self.image_path, 'rb') as f:
                        buf = f.read(512*1024)
                    hay = buf.decode('latin-1', errors='ignore')
                    if ('AI-Interpretation Summary' in hay) or ('LLM Backend' in hay):
                        exists = True
                except Exception:
                    pass
                if not exists:
                    exe = _find_exiftool()
                    if exe:
                        import subprocess, json as _json
                        cmd = [exe, '-EXIF:UserComment', '-IPTC:Keywords', '-IPTC:ObjectName', '-IPTC:Caption-Abstract', '-j', str(self.image_path)]
                        kwargs = _exiftool_run_kwargs(exe, timeout=30)
                        res = subprocess.run(cmd, **kwargs)
                        if res.returncode == 0 and res.stdout.strip():
                            data = _json.loads(res.stdout)
                            if isinstance(data, list) and data:
                                d0 = data[0] if isinstance(data[0], dict) else {}
                                val = d0.get('EXIF:UserComment') or d0.get('UserComment')
                                if val and str(val).strip():
                                    exists = True
                                kw = d0.get('IPTC:Keywords')
                                objn = d0.get('IPTC:ObjectName')
                                cap = d0.get('IPTC:Caption-Abstract')
                                if (kw and str(kw).strip()) or (objn and str(objn).strip()) or (cap and str(cap).strip()):
                                    exists = True
            except Exception:
                exists = False
            self.finished.emit(self.image_path, exists)
    class MetaRunnable(QtCore.QRunnable):
        def __init__(self, task):
            super().__init__()
            self.task = task
            self.setAutoDelete(True)
        def run(self):
            self.task.run()


if __name__ == '__main__':
    main()