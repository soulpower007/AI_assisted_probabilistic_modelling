#!/usr/bin/env python3
"""
Standalone script to fix JSON structure issues in marketing benchmark files.
This script ensures all JSON files have the correct structure with metadata and benchmarks keys.
"""

import json
import os
import glob
from datetime import datetime

def validate_and_fix_json_structure(filename: str) -> bool:
    """
    Validate that a JSON file has the correct structure and fix it if needed.
    Returns True if the file was valid or successfully fixed.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Check if it already has the correct structure
        if "metadata" in data and "benchmarks" in data:
            print(f"✅ {filename} already has correct structure")
            return True
        
        # Check if it has the old structure (direct channel data)
        if "channels" in data and "company" in data:
            print(f"🔄 Fixing structure of {filename}...")
            
            # Create the correct structure
            fixed_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "tool": "elict_priors_gemini",
                    "model": "gemini-2.5-flash",
                    "fixed_at": datetime.now().isoformat()
                },
                "benchmarks": data
            }
            
            # Save the fixed structure
            with open(filename, 'w') as f:
                json.dump(fixed_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Fixed structure of {filename}")
            return True
        
        # Unknown structure
        print(f"❌ {filename} has unknown structure")
        return False
        
    except Exception as e:
        print(f"❌ Error validating {filename}: {e}")
        return False

def fix_all_json_files():
    """Fix all JSON files that don't have the correct structure."""
    print("🔧 Checking and fixing JSON structure in all marketing benchmark files...")
    
    # List of patterns to check
    patterns = [
        "marketing_benchmarks_config.json",
        "marketing_benchmarks_*.json",
        "temp_gemini_results_*.json"
    ]
    
    # Also check for any files that might have been created by the interactive agents
    # These would have the wrong structure (direct channel data without metadata wrapper)
    additional_files = [
        "marketing_benchmarks_config.json"  # This is the main file that gets overwritten
    ]
    
    fixed_count = 0
    total_count = 0
    
    # Check pattern-based files
    for pattern in patterns:
        for filename in glob.glob(pattern):
            total_count += 1
            print(f"\nChecking: {filename}")
            if validate_and_fix_json_structure(filename):
                fixed_count += 1
    
    # Check additional specific files
    for filename in additional_files:
        if os.path.exists(filename):
            total_count += 1
            print(f"\nChecking: {filename}")
            if validate_and_fix_json_structure(filename):
                fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files checked: {total_count}")
    print(f"   Files with correct structure: {fixed_count}")
    print(f"   Files that needed fixing: {total_count - fixed_count}")
    
    if fixed_count == total_count:
        print("✅ All files now have correct structure!")
    else:
        print("⚠️  Some files could not be fixed automatically")
    
    # Additional check for the main config file
    main_file = "marketing_benchmarks_config.json"
    if os.path.exists(main_file):
        print(f"\n🔍 Special check for {main_file}:")
        try:
            with open(main_file, 'r') as f:
                data = json.load(f)
            
            if "metadata" in data and "benchmarks" in data:
                print(f"✅ {main_file} has correct structure")
            else:
                print(f"❌ {main_file} still has wrong structure - attempting to fix...")
                if validate_and_fix_json_structure(main_file):
                    print(f"✅ {main_file} fixed successfully!")
                else:
                    print(f"❌ Failed to fix {main_file}")
        except Exception as e:
            print(f"❌ Error checking {main_file}: {e}")

if __name__ == "__main__":
    fix_all_json_files() 