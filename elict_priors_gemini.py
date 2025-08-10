#!/usr/bin/env python3
"""
Elicit channel priors (CPM, CTR, CVR ranges) using Gemini API

Two modes available:
1. Grounding mode (use_grounding=True): Uses Google Search for better data quality but parses JSON manually
2. Structured mode (use_grounding=False): Uses structured JSON output for guaranteed parsing

Note: Gemini doesn't support both grounding and structured output simultaneously

Env:
  export GEMINI_API_KEY="your_key"

Install:
  pip install -U google-genai pydantic
Docs: https://ai.google.dev/gemini-api/docs/quickstart
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import List, Literal, Optional
from dataclasses import asdict, dataclass
from pydantic import BaseModel, Field, model_validator
from google import genai
from google.genai import types

# --------- Schema for structured output ---------
class Range(BaseModel):
    unit: Literal["USD", "ratio", "per_1000_impressions"] = Field(..., description="Unit for the metric")
    p10: float = Field(..., description="10th percentile (lower bound)")
    p50: float = Field(..., description="Median (typical)")
    p90: float = Field(..., description="90th percentile (upper bound)")
    
    @model_validator(mode='after')
    def validate_percentiles(self):
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(f"Invalid range: p10={self.p10}, p50={self.p50}, p90={self.p90}. Must have p10 <= p50 <= p90")
        return self

class Source(BaseModel):
    title: str = Field(..., description="Source title or description")
    uri: str = Field(..., description="Web URL of the source")

class ChannelBenchmarks(BaseModel):
    channel: str = Field(..., description="Marketing channel name")
    geography: Optional[str] = Field(None, description="Country/region the estimates apply to")
    industry: Optional[str] = Field(None, description="Industry or business model context")
    cpm_usd_per_1000: Range = Field(..., description="CPM in USD per 1000 impressions")
    ctr_ratio: Range = Field(..., description="CTR as fraction 0..1")
    cvr_ratio: Range = Field(..., description="CVR as fraction 0..1")
    aov_usd: Range = Field(..., description="Average Order Value in USD")
    notes: Optional[str] = Field(None, description="Short caveats (auction variation, seasonality, etc.)")
    sources: Optional[List[Source]] = Field(None, description="Web sources used for this channel's benchmarks")

class BenchmarksResponse(BaseModel):
    company: str
    industry: Optional[str]
    geography: Optional[str]
    timeframe: Optional[str]
    channels: List[ChannelBenchmarks]

# --------- Response transformation helper ---------
def transform_legacy_response(json_data: dict, company: str, industry: Optional[str], geography: Optional[str], channels: List[str]) -> dict:
    """
    Transform legacy response format (channel names as keys) to our expected schema format.
    """
    transformed_channels = []
    
    # Map common channel name variations
    channel_mapping = {
        "google_ads": "Google",
        "google": "Google", 
        "meta_ads": "Meta",
        "meta": "Meta",
        "facebook": "Meta",
        "tiktok_ads": "TikTok",
        "tiktok": "TikTok",
        "linkedin_ads": "LinkedIn",
        "linkedin": "LinkedIn"
    }
    
    for key, value in json_data.items():
        if isinstance(value, dict) and any(metric in value for metric in ["cpm_usd", "ctr", "cvr"]):
            # This looks like channel data
            channel_name = channel_mapping.get(key.lower(), key.title())
            
            # Transform the metrics to our expected format
            channel_data = {
                "channel": channel_name,
                "geography": geography,
                "industry": industry,
                "notes": value.get("notes", ""),
                "sources": value.get("sources", [])
            }
            
            # Transform CPM
            if "cpm_usd" in value:
                cpm = value["cpm_usd"]
                channel_data["cpm_usd_per_1000"] = {
                    "unit": "USD",
                    "p10": float(cpm.get("p10", 0)),
                    "p50": float(cpm.get("p50", 0)),
                    "p90": float(cpm.get("p90", 0))
                }
            
            # Transform CTR
            if "ctr" in value:
                ctr = value["ctr"]
                channel_data["ctr_ratio"] = {
                    "unit": "ratio",
                    "p10": float(ctr.get("p10", 0)),
                    "p50": float(ctr.get("p50", 0)),
                    "p90": float(ctr.get("p90", 0))
                }
            
            # Transform CVR
            if "cvr" in value:
                cvr = value["cvr"]
                channel_data["cvr_ratio"] = {
                    "unit": "ratio", 
                    "p10": float(cvr.get("p10", 0)),
                    "p50": float(cvr.get("p50", 0)),
                    "p90": float(cvr.get("p90", 0))
                }
            
            # Transform AOV
            if "aov" in value or "aov_usd" in value:
                aov = value.get("aov", value.get("aov_usd", {}))
                channel_data["aov_usd"] = {
                    "unit": "USD",
                    "p10": float(aov.get("p10", 100)),
                    "p50": float(aov.get("p50", 150)),
                    "p90": float(aov.get("p90", 250))
                }
            else:
                # Default AOV if not provided
                channel_data["aov_usd"] = {
                    "unit": "USD",
                    "p10": 100.0,
                    "p50": 150.0,
                    "p90": 250.0
                }
            
            transformed_channels.append(channel_data)
    
    return {
        "company": company,
        "industry": industry,
        "geography": geography,
        "timeframe": "2023-2024",
        "channels": transformed_channels
    }

# --------- Prompt helper ---------
def build_prompt(company: str, industry: str | None, geography: str | None, channels: List[str], use_grounding: bool = True, initial_message: str | None = None) -> str:
    chans = ", ".join(channels)
    json_instruction = "Return *only* valid JSON" if use_grounding else "Return *only* JSON (the API constrains this)"
    
    # Add initial message context if provided
    initial_context = ""
    if initial_message:
        initial_context = f"""
User's Initial Request: "{initial_message}"

Please use this context to better understand what the user is looking for and tailor your response accordingly.
"""
    
    return f"""
You are a marketing analyst. For the company "{company}", elicit realistic *benchmark ranges* for ad performance metrics
for channels: {chans}. If not specified, assume B2B SaaS-like funnel, paid social + search mix.

{initial_context}
{json_instruction} with the following exact structure:

{{
  "company": "{company}",
  "industry": "{industry or "unspecified"}",
  "geography": "{geography or "unspecified"}",
  "timeframe": "2023-2024",
  "channels": [
    {{
      "channel": "Google",
      "geography": "{geography or "unspecified"}",
      "industry": "{industry or "unspecified"}",
      "cpm_usd_per_1000": {{
        "unit": "USD",
        "p10": 25.0,
        "p50": 45.0,
        "p90": 75.0
      }},
      "ctr_ratio": {{
        "unit": "ratio",
        "p10": 0.015,
        "p50": 0.035,
        "p90": 0.065
      }},
      "cvr_ratio": {{
        "unit": "ratio",
        "p10": 0.01,
        "p50": 0.025,
        "p90": 0.05
      }},
      "aov_usd": {{
        "unit": "USD",
        "p10": 80.0,
        "p50": 150.0,
        "p90": 300.0
      }},
      "notes": "Search ads typically have higher CTR/CVR than display",
      "sources": [
        {{
          "title": "Google Ads Benchmarks 2024",
          "uri": "https://ads.google.com/research/benchmarks"
        }},
        {{
          "title": "HubSpot Marketing Report",
          "uri": "https://blog.hubspot.com/marketing/benchmarks"
        }}
      ]
    }}
  ]
}}

Guidelines:
- Base all numeric ranges on *current* public web sources via Google Search (grounding enabled).
- Search for recent benchmark reports, industry studies, and marketing research from 2023-2024.
- Prefer authoritative sources: marketing platforms (Google Ads, Meta Business), research firms (HubSpot, Salesforce), industry reports (eMarketer, Statista).
- Avoid forum posts, outdated data, or unreliable sources.
- Keep CTR/CVR as fractions (e.g., 0.012 not 1.2%).
- AOV should reflect typical purchase values for the industry/geography.
- Add a short 'notes' field per channel with caveats: seasonality, audience quality, lead intent.
- Use the provided geography and industry as filters if present.
- Include ALL requested channels: {chans}
- Ensure web sources are cited and accessible for verification.

Context:
- Company: {company}
- Industry: {industry or "unspecified"}
- Geography: {geography or "unspecified"}
- Timeframe: Prefer the most recent 12–24 months

Metrics to include:
- CPM in USD per 1000 impressions
- CTR (Click-Through Rate) as fraction 0..1
- CVR (Conversion Rate) as fraction 0..1  
- AOV (Average Order Value) in USD
""".strip()

# --------- Core function ---------
def elicit_priors_with_gemini(
    company: str,
    industry: Optional[str] = None,
    geography: Optional[str] = None,
    channels: Optional[List[str]] = None,
    model_name: str = "gemini-2.5-flash",  # fast + grounding support
    use_grounding: bool = True,  # Whether to use Google Search grounding
    initial_message: Optional[str] = None,  # Initial user message for context
) -> BenchmarksResponse:
    """
    Returns:
      - BenchmarksResponse (parsed JSON with ranges)
    """
    # Validate API key
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    if channels is None:
        channels = ["Google", "Meta", "TikTok", "LinkedIn"]

    try:
        client = genai.Client()
        print("Client initialized")
        print(f"Model: {model_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    # Configure based on whether we want grounding or structured output
    # Note: Gemini doesn't support both tools and structured JSON output simultaneously
    if use_grounding:
        # Use Google Search grounding but no structured output
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            # No response_mime_type or response_schema when using tools
        )
        print("Using grounding mode (no structured output)")
    else:
        # Use structured JSON output but no grounding
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BenchmarksResponse,  # Pydantic schema -> constrained JSON
        )
        print("Using structured output mode (no grounding)")
    print("Config initialized")
    prompt = build_prompt(company, industry, geography, channels, use_grounding, initial_message)
    # print(f"Prompt built: {prompt[:2000]}...")  # Truncate for readability
    try:

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        print(f"Response type: {type(response)}")
        print(f"Response has parsed: {hasattr(response, 'parsed')}")
        if hasattr(response, 'text'):
            print(f"Response text: {response.text[:500]}...")

        # Handle response based on mode
        if use_grounding:
            # Parse JSON manually from text response
            import json
            try:
                if hasattr(response, 'text') and response.text:
                    # Extract JSON from the response text
                    response_text = response.text.strip()
                    # Sometimes the response has markdown formatting, so clean it
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    json_data = json.loads(response_text)
                    
                    # Check if the response matches our expected structure
                    if "company" in json_data and "channels" in json_data:
                        # Direct parsing - matches our schema
                        data = BenchmarksResponse(**json_data)
                        print("Successfully parsed structured JSON from grounding response")
                    else:
                        # Fallback: transform the response to match our schema
                        print("Transforming non-standard JSON structure to match schema")
                        transformed_data = transform_legacy_response(json_data, company, industry, geography, channels)
                        data = BenchmarksResponse(**transformed_data)
                        print("Successfully transformed and parsed JSON from grounding response")
                else:
                    raise RuntimeError("No text response received")
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(f"Failed to parse JSON from grounding response: {e}. Response: {response.text[:1000]}")
        else:
            # Use structured output parsing
            if hasattr(response, 'parsed'):
                print(f"Response parsed: {response.parsed}")
            if not hasattr(response, 'parsed') or not response.parsed:
                error_msg = "Gemini did not return parsed JSON."
                if hasattr(response, 'text'):
                    error_msg += f" Raw response: {response.text}"
                raise RuntimeError(error_msg)
            data: BenchmarksResponse = response.parsed

        return data
    
    except Exception as e:
        raise RuntimeError(f"Failed to elicit priors: {e}")

# --------- Pretty printing ---------
def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

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

def save_results_to_json(result: BenchmarksResponse, filename: str = None) -> str:
    """Save the benchmark results to a JSON file."""
    if filename is None:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"marketing_benchmarks_config.json"
    
    # Validate the result before saving
    if not isinstance(result, BenchmarksResponse):
        raise ValueError(f"Expected BenchmarksResponse, got {type(result)}")
    
    # Convert Pydantic model to dict
    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "tool": "elict_priors_gemini",
            "model": "gemini-2.5-flash"
        },
        "benchmarks": result.model_dump()
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Verify the saved file has correct structure
    if not validate_and_fix_json_structure(filename):
        raise RuntimeError(f"Failed to save {filename} with correct structure")
    
    print(f"✅ Results saved to: {filename}")
    return filename

def ensure_json_structure(filename: str) -> bool:
    """
    Ensure that a JSON file has the correct structure before processing.
    This is called before any operations that expect the benchmarks structure.
    """
    if not os.path.exists(filename):
        print(f"⚠️  {filename} does not exist")
        return False
    
    return validate_and_fix_json_structure(filename)

def load_results_from_json(filename: str) -> BenchmarksResponse:
    """Load benchmark results from a JSON file."""
    # Ensure the file has correct structure before loading
    if not ensure_json_structure(filename):
        raise RuntimeError(f"Cannot load {filename} - invalid structure")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract benchmarks
    benchmarks = BenchmarksResponse(**data["benchmarks"])
    
    print(f"✅ Results loaded from: {filename}")
    print(f"📊 Company: {benchmarks.company}")
    
    return benchmarks

def print_benchmarks(result: BenchmarksResponse):
    print(f"\nCompany: {result.company}")
    if result.industry: print(f"Industry: {result.industry}")
    if result.geography: print(f"Geography: {result.geography}")
    if result.timeframe: print(f"Timeframe: {result.timeframe}")

    for ch in result.channels:
        print(f"\n[{ch.channel}]")
        print(f"  CPM (USD per 1000): p10={ch.cpm_usd_per_1000.p10:.2f}, "
              f"p50={ch.cpm_usd_per_1000.p50:.2f}, p90={ch.cpm_usd_per_1000.p90:.2f}")
        print(f"  CTR (ratio):        p10={fmt_pct(ch.ctr_ratio.p10)}, "
              f"p50={fmt_pct(ch.ctr_ratio.p50)}, p90={fmt_pct(ch.ctr_ratio.p90)}")
        print(f"  CVR (ratio):        p10={fmt_pct(ch.cvr_ratio.p10)}, "
              f"p50={fmt_pct(ch.cvr_ratio.p50)}, p90={fmt_pct(ch.cvr_ratio.p90)}")
        print(f"  AOV (USD):          p10=${ch.aov_usd.p10:.2f}, "
              f"p50=${ch.aov_usd.p50:.2f}, p90=${ch.aov_usd.p90:.2f}")
        if ch.notes:
            print(f"  Notes: {ch.notes}")
        if ch.sources:
            print(f"  Sources:")
            for i, source in enumerate(ch.sources, 1):
                print(f"    [{i}] {source.title} — {source.uri}")

# --------- Example usage ---------
def test_json_structure_validation():
    """Test the JSON structure validation and fixing functionality."""
    print("\n🧪 Testing JSON structure validation...")
    
    # Test with current file
    current_file = "marketing_benchmarks_config.json"
    if os.path.exists(current_file):
        print(f"Testing current file: {current_file}")
        validate_and_fix_json_structure(current_file)
    
    # Test with a malformed file
    test_file = "test_malformed.json"
    test_data = {
        "company": "Test Company",
        "channels": [{"channel": "Test"}]
    }
    
    # Create a malformed file
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nTesting malformed file: {test_file}")
    validate_and_fix_json_structure(test_file)
    
    # Check if it was fixed
    with open(test_file, 'r') as f:
        fixed_data = json.load(f)
    
    if "metadata" in fixed_data and "benchmarks" in fixed_data:
        print("✅ Malformed file was successfully fixed!")
    else:
        print("❌ Malformed file was not fixed properly")
    
    # Clean up test file
    os.remove(test_file)
    print("🧹 Test file cleaned up")

def fix_existing_json_files():
    """Fix any existing JSON files that don't have the correct structure."""
    print("\n🔧 Checking and fixing existing JSON files...")
    
    # List of files to check
    files_to_check = [
        "marketing_benchmarks_config.json",
        "marketing_benchmarks_*.json"
    ]
    
    import glob
    for pattern in files_to_check:
        for filename in glob.glob(pattern):
            print(f"Checking: {filename}")
            validate_and_fix_json_structure(filename)
    
    print("✅ JSON file structure check complete")

if __name__ == "__main__":
    # Fix any existing JSON files first
    fix_existing_json_files()
    
    # Test JSON structure validation
    test_json_structure_validation()
    
    # Example: swap these for your hackathon inputs
    company = "Example B2B SaaS"
    industry = "B2B SaaS"
    geography = "United States"
    
    # Test transformation function first 
    test_legacy_data = {
        "google_ads": {
            "cpm_usd": {"p10": 40.0, "p50": 55.0, "p90": 75.0},
            "ctr": {"p10": 0.02, "p50": 0.045, "p90": 0.075},
            "cvr": {"p10": 0.01, "p50": 0.02, "p90": 0.04},
            "aov": {"p10": 120.0, "p50": 180.0, "p90": 280.0},
            "notes": "Google Ads test"
        }
    }
    
    print("Testing transformation function...")
    try:
        transformed = transform_legacy_response(test_legacy_data, company, industry, geography, ["Google"])
        test_result = BenchmarksResponse(**transformed)
        print("✅ Transformation function works!")
        print(f"Transformed channel: {test_result.channels[0].channel}")
    except Exception as e:
        print(f"❌ Transformation function failed: {e}")
        
    print("\nStarting Gemini API call...")
    result = elicit_priors_with_gemini(
        company=company,
        industry=industry,
        geography=geography,
        channels=["Google", "Meta", "TikTok", "LinkedIn"],
        model_name="gemini-2.5-flash",
        use_grounding=True,  # Set to False for structured output mode
    )
    
    # Print results to console
    print_benchmarks(result)
    
    # Save results to JSON file with validation
    try:
        filename = save_results_to_json(result)
        print(f"✅ Results successfully saved with correct structure")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        # Try to save with a backup filename
        backup_filename = f"marketing_benchmarks_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            # Direct save without validation as fallback
            data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "tool": "elict_priors_gemini",
                    "model": "gemini-2.5-flash",
                    "note": "backup_save_due_to_error"
                },
                "benchmarks": result.model_dump()
            }
            with open(backup_filename, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"⚠️  Saved backup to: {backup_filename}")
            filename = backup_filename
        except Exception as backup_e:
            print(f"❌ Backup save also failed: {backup_e}")
            raise RuntimeError(f"Failed to save results: {e}")
    
    # Auto-convert to channel priors if the conversion script exists
    try:
        import subprocess
        conversion_script = "convert_benchmarks_to_priors.py"
        if os.path.exists(conversion_script):
            print(f"\n🔄 Auto-converting to channel priors...")
            subprocess.run(["python", conversion_script], check=True)
            print(f"✅ Channel priors updated in config/channel_priors.json")
    except Exception as e:
        print(f"⚠️  Auto-conversion failed: {e}")
        print(f"💡 You can manually run: python convert_benchmarks_to_priors.py")
    
    print(f"\n📊 Marketing benchmarks successfully generated!")
    print(f"📁 Saved to: {filename}")

