#!/usr/bin/env python3
"""
Race Metadata Parser - Extract race data from timing PDFs

Supports multiple timing system formats:
- TSL Timing (CLASSIFICATION/SESSION keywords)
- S.M.A.R.T Timing (Scottish Motorsports Automatic Race Timing)
- Auto-detects format and routes to appropriate parser
"""

import re
from datetime import datetime
from typing import Dict, Tuple, Optional
import pdfplumber


def detect_pdf_format(pdf_path: str) -> str:
    """
    Detect the timing PDF format by examining the first few pages.
    
    Returns:
        str: Format identifier ('tsl', 'smart', or 'unknown')
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check first few pages for format indicators
            text_samples = []
            for i, page in enumerate(pdf.pages[:5]):
                text = page.extract_text()
                if text:
                    text_samples.append(text.upper())
            
            combined_text = ' '.join(text_samples)
            
            # Check for S.M.A.R.T indicators
            if any(indicator in combined_text for indicator in [
                'S.M.A.R.T', 'SMART TIMING', 'SCOTTISH MOTORSPORTS AUTOMATIC RACE TIMING',
                'SMART-TIMING.CO.UK', 'MYLAPS.COM'
            ]):
                return 'smart'
            
            # Check for TSL indicators
            if any(indicator in combined_text for indicator in [
                'CLASSIFICATION', 'SESSION', 'TSL'
            ]):
                return 'tsl'
            
            # Check for common race timing patterns
            if 'RACE' in combined_text and ('POS' in combined_text or 'NO.' in combined_text):
                # Likely a timing PDF, default to S.M.A.R.T if no TSL indicators
                return 'smart'
            
            return 'unknown'
    except Exception as e:
        print(f"Error detecting PDF format: {e}")
        return 'unknown'


def extract_track_and_event(pdf_path: str) -> Tuple[str, str]:
    """
    Extract track/venue name and event name from PDF.
    
    Returns:
        tuple: (track_name, event_name)
    """
    track_name = ''
    event_name = ''
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check first few pages for track/event info
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                # Look for common patterns:
                # - Track names often appear in headers/footers
                # - Event names might be in title or header
                # - Common patterns: "At [Track Name]", "[Track Name] Circuit", etc.
                
                # Try to find track name patterns
                track_patterns = [
                    r'at\s+([A-Z][A-Za-z\s&]+?)(?:\s+Circuit|\s+Raceway|\s+Track|\s+Speedway|$)',
                    r'([A-Z][A-Za-z\s&]+?)\s+Circuit',
                    r'([A-Z][A-Za-z\s&]+?)\s+Raceway',
                    r'([A-Z][A-Za-z\s&]+?)\s+Speedway',
                    r'Venue:\s*([A-Z][A-Za-z\s&]+)',
                    r'Track:\s*([A-Z][A-Za-z\s&]+)',
                    r'Location:\s*([A-Z][A-Za-z\s&]+)',
                ]
                
                for pattern in track_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        # Filter out common false positives
                        if len(candidate) > 3 and candidate.lower() not in ['the', 'and', 'for', 'with', 'from']:
                            track_name = candidate
                            break
                
                # Try to find event name patterns
                event_patterns = [
                    r'([A-Z][A-Za-z\s&\-]+?)\s+Championship',
                    r'([A-Z][A-Za-z\s&\-]+?)\s+Championships',
                    r'([A-Z][A-Za-z\s&\-]+?)\s+Series',
                    r'([A-Z][A-Za-z\s&\-]+?)\s+Event',
                    r'Event:\s*([A-Z][A-Za-z\s&\-]+)',
                    r'Championship:\s*([A-Z][A-Za-z\s&\-]+)',
                ]
                
                for pattern in event_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        if len(candidate) > 3:
                            event_name = candidate
                            break
                
                # If we found both, we can stop
                if track_name and event_name:
                    break
                
                # Also check for track name in filename or path
                if not track_name:
                    # Common UK tracks
                    uk_tracks = ['Lydden Hill', 'Croft', 'Mondello Park', 'Pembrey', 'Anglesey', 
                                'Knockhill', 'Brands Hatch', 'Silverstone', 'Donington', 'Oulton Park',
                                'Cadwell Park', 'Snetterton', 'Thruxton', 'Rockingham']
                    for track in uk_tracks:
                        if track.lower() in text.lower():
                            track_name = track
                            break
    except Exception as e:
        print(f"[PDF-EXTRACT] Error extracting track/event: {e}")
    
    return track_name, event_name


def parse_timing_pdf(pdf_path: str) -> Tuple[Dict, Dict, str, str]:
    """
    Main parsing function for timing PDFs. Auto-detects format and routes to appropriate parser.
    
    Returns:
        tuple: (race_data, sessions, track_name, event_name)
            - race_data: {race_number: {car, team, driver, session, time_start, time_end}}
            - sessions: {session_name: (start_datetime, end_datetime)}
            - track_name: Track/venue name (if found)
            - event_name: Event/championship name (if found)
    """
    # Extract track and event name
    track_name, event_name = extract_track_and_event(pdf_path)
    
    # Detect format
    format_type = detect_pdf_format(pdf_path)
    print(f"[PDF-DETECT] Detected format: {format_type}")
    
    if format_type == 'tsl':
        race_data, sessions = parse_tsl_pdf(pdf_path)
        return race_data, sessions, track_name, event_name
    elif format_type == 'smart':
        race_data, sessions = parse_smart_pdf(pdf_path)
        return race_data, sessions, track_name, event_name
    else:
        print(f"[PDF-ERROR] Unknown or unsupported PDF format")
        # Try TSL first as fallback
        race_data, sessions = parse_tsl_pdf(pdf_path)
        return race_data, sessions, track_name, event_name


def parse_tsl_pdf(pdf_path: str) -> Tuple[Dict, Dict]:
    """
    Parse TSL timing PDF format (original parser).
    """
    race_data = {}
    sessions = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract session info from all pages
            sessions = extract_session_info_from_pdf(pdf)
            
            # Extract race data from classification pages
            race_data = extract_race_data_from_pdf(pdf)
            
    except Exception as e:
        print(f"Error parsing TSL PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return race_data, sessions


def parse_smart_pdf(pdf_path: str) -> Tuple[Dict, Dict]:
    """
    Parse S.M.A.R.T timing PDF format.
    
    Format characteristics:
    - Header: "Race X - [Category] - [Type] [Date] [Time]"
    - Classification: "Pos No. Name Make/Model CC Class Laps Total Tm Diff Best Tm In Lap 2nd Best"
    - Session start: "Race (X Laps) started at HH:MM:SS"
    """
    race_data = {}
    sessions = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            current_session = None
            current_race_start = None
            
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                # Look for race header: "Race X - [Category] - [Type] [Date] [Time]"
                # Example: "Race 1 - Juniors - Heat 1 - Re-Start 19/10/2025 09:38"
                race_header_match = re.search(
                    r'Race\s+(\d+)\s*-\s*([^-]+?)\s*-\s*([^-]+?)\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})',
                    text,
                    re.IGNORECASE
                )
                
                if race_header_match:
                    race_num = race_header_match.group(1)
                    category = race_header_match.group(2).strip()
                    race_type = race_header_match.group(3).strip()
                    date_str = race_header_match.group(4)
                    time_str = race_header_match.group(5)
                    
                    # Build session name
                    current_session = f"{category} - {race_type}"
                    
                    # Parse date and time for session
                    try:
                        day, month, year = date_str.split('/')
                        hour, minute = time_str.split(':')
                        session_start = datetime(
                            int(year), int(month), int(day),
                            int(hour), int(minute), 0
                        )
                        
                        # Look for race start time: "Race (X Laps) started at HH:MM:SS"
                        race_start_match = re.search(
                            r'Race\s*\([^)]+\)\s+started\s+at\s+(\d{1,2}):(\d{2}):(\d{2})',
                            text,
                            re.IGNORECASE
                        )
                        
                        if race_start_match:
                            start_h = int(race_start_match.group(1))
                            start_m = int(race_start_match.group(2))
                            start_s = int(race_start_match.group(3))
                            current_race_start = datetime(
                                int(year), int(month), int(day),
                                start_h, start_m, start_s
                            )
                        else:
                            current_race_start = session_start
                        
                        # Estimate session end (add 2 hours as default)
                        session_end = session_start.replace(hour=(session_start.hour + 2) % 24)
                        
                        sessions[current_session] = (session_start, session_end)
                        print(f"[SMART-PARSE] Found session: {current_session} at {session_start}")
                        
                    except Exception as e:
                        print(f"[SMART-PARSE] Error parsing session time: {e}")
                
                # Look for classification table header
                # "Pos No. Name Make/Model CC Class Laps Total Tm Diff Best Tm In Lap 2nd Best"
                if 'Pos No. Name Make/Model' in text or 'Pos No. Name' in text:
                    # Parse classification entries
                    i = 0
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Skip header lines
                        if 'Pos No. Name' in line or 'Sorted on Laps' in line or not line:
                            i += 1
                            continue
                        
                        # Skip non-data lines
                        if any(skip in line for skip in [
                            'Margin of Victory', 'Senior Clerk', 'Chief Timekeeper',
                            'Results available', 'Printed:', 'Announcements', 'Not classified',
                            'BTRDA', 'All Classes', 'Race ('
                        ]):
                            i += 1
                            continue
                        
                        # Parse entry line: "1 999 Hayden HARRISS Suzuki Swift 1300 J 4 3:43.188 54.197 3 54.450"
                        # Format: POS NO NAME MAKE/MODEL CC CLASS LAPS TOTAL_TM DIFF BEST_TM IN_LAP 2ND_BEST
                        # Check if line starts with position number
                        if re.match(r'^\d+\s+\d+', line):
                            try:
                                # Split line into parts
                                parts = line.split()
                                if len(parts) < 8:  # Need at least POS NO NAME MAKE MODEL CC CLASS LAPS TIME
                                    i += 1
                                    continue
                                
                                pos = parts[0]
                                race_number = parts[1]
                                
                                # Find CC index (first number after name/make/model)
                                cc_idx = None
                                for idx, part in enumerate(parts[2:], start=2):
                                    if re.match(r'^\d+[A-Z]?$', part):  # CC like "1300" or "2000T"
                                        cc_idx = idx
                                        break
                                
                                if not cc_idx or cc_idx < 3:
                                    i += 1
                                    continue
                                
                                # Name is parts[2] to before make/model
                                # Make/model starts where we find a brand name
                                brands = ['Suzuki', 'BMW', 'Nissan', 'Ford', 'Vauxhall', 'Mercedes', 
                                        'Honda', 'Alfa', 'Renault', 'Citroen', 'Peugeot', 'MG', 
                                        'FIA', 'MV', 'BMC', 'Classic', 'Mercedes']
                                
                                make_model_start = None
                                for idx in range(2, cc_idx):
                                    if any(brand.lower() in parts[idx].lower() for brand in brands):
                                        make_model_start = idx
                                        break
                                
                                if make_model_start:
                                    name_parts = parts[2:make_model_start]
                                    make_model_parts = parts[make_model_start:cc_idx]
                                else:
                                    # Fallback: assume name is 1-2 words
                                    name_parts = parts[2:min(4, cc_idx)]
                                    make_model_parts = parts[len(name_parts)+2:cc_idx]
                                
                                driver_name = ' '.join(name_parts).strip()
                                car = ' '.join(make_model_parts).strip()
                                
                                if driver_name and car and race_number not in race_data:
                                    race_data[race_number] = {
                                        'car': car,
                                        'team': '',
                                        'driver': driver_name,
                                        'session': current_session,
                                        'race_start': current_race_start.isoformat() if current_race_start else None,
                                    }
                                    print(f"[SMART-PARSE] Extracted: {race_number} -> car={car}, driver={driver_name}")
                                
                            except Exception as e:
                                print(f"[SMART-PARSE] Error parsing entry: {line[:100]} - {e}")
                        
                        # Also check for DNS/DNF entries
                        elif re.match(r'^(DNS|DNF)\s+(\d+)', line):
                            dns_match = re.match(r'^(DNS|DNF)\s+(\d+)\s+(.+)', line)
                            if dns_match:
                                race_number = dns_match.group(2)
                                car_info = dns_match.group(3).strip()
                                
                                # Extract make/model from car_info
                                # Format: "Oscar COOPER Suzuki Swift 1300 J DNS 0"
                                car_match = re.search(r'([A-Z][A-Za-z0-9\s/]+?)\s+\d+', car_info)
                                car = car_match.group(1).strip() if car_match else car_info
                                
                                if race_number not in race_data:
                                    race_data[race_number] = {
                                        'car': car,
                                        'team': '',
                                        'driver': '',
                                        'session': current_session,
                                        'status': dns_match.group(1),
                                    }
                        
                        i += 1
                
    except Exception as e:
        print(f"Error parsing S.M.A.R.T PDF: {e}")
        import traceback
        traceback.print_exc()
    
    return race_data, sessions


def extract_session_info_from_pdf(pdf) -> Dict:
    """
    Extract session names and time ranges from PDF footers/headers (TSL format).
    
    Returns:
        dict: {session_name: (start_datetime, end_datetime)}
    """
    sessions = {}
    
    try:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            # Look for session headers like "TEST SESSION 1 - CLASSIFICATION"
            session_match = re.search(r'([A-Z\s]+SESSION\s+\d+)\s*-\s*([A-Z]+)', text, re.IGNORECASE)
            if session_match:
                session_name = session_match.group(1).strip()
                
                # Look for date and time in footer
                # Pattern: "Date: DD/MM/YYYY Start: HH:MM Finish: HH:MM"
                date_match = re.search(r'Date:\s*(\d{2}/\d{2}/\d{4})', text)
                start_match = re.search(r'Start:\s*(\d{2}:\d{2})', text)
                finish_match = re.search(r'Finish:\s*(\d{2}:\d{2})', text)
                
                if date_match and start_match and finish_match:
                    date_str = date_match.group(1)
                    start_time = start_match.group(1)
                    finish_time = finish_match.group(1)
                    
                    try:
                        # Parse date: DD/MM/YYYY
                        day, month, year = date_str.split('/')
                        # Parse start time: HH:MM
                        start_h, start_m = start_time.split(':')
                        # Parse finish time: HH:MM
                        finish_h, finish_m = finish_time.split(':')
                        
                        start_datetime = datetime(
                            int(year), int(month), int(day),
                            int(start_h), int(start_m), 0
                        )
                        end_datetime = datetime(
                            int(year), int(month), int(day),
                            int(finish_h), int(finish_m), 0
                        )
                        
                        sessions[session_name] = (start_datetime, end_datetime)
                    except Exception as e:
                        print(f"Error parsing session time for {session_name}: {e}")
                        continue
    
    except Exception as e:
        print(f"Error extracting session info: {e}")
    
    return sessions


def extract_race_data_from_pdf(pdf) -> Dict:
    """
    Extract race number → car/team/driver mappings from classification pages (TSL format).
    
    Returns:
        dict: {race_number: {car, team, driver, session, time_start, time_end}}
    """
    race_data = {}
    current_session = None
    
    try:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            # Check if this is a classification page
            if 'CLASSIFICATION' in text.upper():
                # Extract session name
                session_match = re.search(r'([A-Z\s]+SESSION\s+\d+)\s*-\s*CLASSIFICATION', text, re.IGNORECASE)
                if session_match:
                    current_session = session_match.group(1).strip()
                    print(f"[PDF-PARSE] Found session: {current_session}")
                
                # Parse classification table
                lines = text.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip header line and empty lines
                    if 'POS NO CL' in line.upper() or 'TEAM / DRIVERS' in line.upper() or not line:
                        i += 1
                        continue
                    
                    # Look for car/team line (contains car brand or model)
                    if any(brand in line for brand in ['Porsche', 'Ferrari', 'BMW', 'Ginetta', 'Aston', 'Lamborghini', 'McLaren', 'Mercedes', 'Audi', 'Cup', 'Challenge', 'GT', 'Supercup']):
                        car_line = line
                        # Next line should be the classification entry
                        if i + 1 < len(lines):
                            entry_line = lines[i + 1].strip()
                            
                            # Check if next line looks like an entry (starts with position number)
                            if re.match(r'^\d+\s+\d+', entry_line):
                                # Parse entry
                                parts = entry_line.split()
                                if len(parts) >= 2:
                                    try:
                                        race_number = parts[1].strip()
                                        
                                        # Extract team/driver from entry line
                                        if len(parts) >= 5:
                                            team_driver_parts = []
                                            for j in range(4, len(parts)):
                                                part = parts[j]
                                                if re.match(r'^\d+\.\d+', part):
                                                    break
                                                team_driver_parts.append(part)
                                            
                                            team_driver_part = ' '.join(team_driver_parts).strip()
                                            
                                            # Split team and drivers
                                            if '/' in team_driver_part:
                                                parts_split = [p.strip() for p in team_driver_part.split('/')]
                                                if len(parts_split) >= 2:
                                                    team = parts_split[0].strip()
                                                    drivers = '/'.join(parts_split[1:]).strip()
                                                else:
                                                    team = ''
                                                    drivers = team_driver_part
                                            else:
                                                if team_driver_part.isupper() or (len(team_driver_part.split()) == 1 and team_driver_part[0].isupper()):
                                                    team = team_driver_part
                                                    drivers = ''
                                                else:
                                                    team = ''
                                                    drivers = team_driver_part
                                            
                                            car = car_line.strip()
                                            
                                            # Try to extract team from car_line if not found
                                            if not team and car:
                                                team_match = re.match(r'^([A-Z][A-Za-z\s&]+?)\s+(Porsche|Ferrari|BMW|Ginetta|Aston|Lamborghini|McLaren|Mercedes|Audi)', car)
                                                if team_match:
                                                    team = team_match.group(1).strip()
                                            
                                            # Store race data
                                            race_data[race_number] = {
                                                'car': car,
                                                'team': team,
                                                'driver': drivers,
                                                'session': current_session,
                                            }
                                            print(f"[PDF-PARSE] Extracted: {race_number} -> car={car}, team={team}, driver={drivers}")
                                    except Exception as e:
                                        print(f"[PDF-PARSE] Error parsing entry line: {entry_line[:100]} - {e}")
                                        pass
                    
                    i += 1
            
            # Also check sector analysis pages
            elif 'SECTOR ANALYSIS' in text.upper():
                session_match = re.search(r'([A-Z\s]+SESSION\s+\d+)\s*-\s*SECTOR\s+ANALYSIS', text, re.IGNORECASE)
                if session_match:
                    current_session = session_match.group(1).strip()
                
                lines = text.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    pos_match = re.match(r'P\d+\s+(\d+)\s+[A-Z]+\s+(.+)', line)
                    if pos_match:
                        race_number = pos_match.group(1).strip()
                        car_team = pos_match.group(2).strip()
                        
                        if i + 1 < len(lines):
                            driver_line = lines[i + 1].strip()
                            driver_match = re.search(r'D1:([^D]+)(?:D2:(.+))?', driver_line)
                            
                            if driver_match:
                                driver1 = driver_match.group(1).strip()
                                driver2 = driver_match.group(2).strip() if driver_match.group(2) else ''
                                drivers = f"{driver1}/{driver2}" if driver2 else driver1
                                
                                if race_number not in race_data:
                                    race_data[race_number] = {}
                                
                                race_data[race_number].update({
                                    'car': car_team,
                                    'driver': drivers,
                                    'session': current_session,
                                })
                                
                                team_match = re.match(r'^([A-Z][A-Za-z\s&]+?)\s+(Porsche|Ferrari|BMW|Ginetta|Aston|Lamborghini|McLaren|Mercedes|Audi)', car_team)
                                if team_match:
                                    race_data[race_number]['team'] = team_match.group(1).strip()
                    
                    i += 1
    
    except Exception as e:
        print(f"Error extracting race data: {e}")
        import traceback
        traceback.print_exc()
    
    return race_data


def extract_tables_from_pdf(pdf_path: str) -> list:
    """
    Extract tables from PDF using pdfplumber.
    
    Returns:
        list: List of extracted tables
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
    except Exception as e:
        print(f"Error extracting tables: {e}")
    return tables


def parse_tsl_timing_format(text: str, tables: list) -> Dict:
    """
    TSL-specific parser for timing format.
    
    This is a fallback parser that can be enhanced for specific TSL formats.
    """
    race_data = {}
    
    # This can be enhanced with TSL-specific patterns
    # For now, rely on the main extraction functions
    
    return race_data
