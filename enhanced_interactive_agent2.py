#!/usr/bin/env python3
"""
Enhanced Interactive Marketing Agent with Gemini Integration

This version actually uses Gemini API for parameter extraction instead of just showing the prompt.
It now integrates directly with the marketing chatbot instead of using system calls.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Import the Gemini function
from elict_priors_gemini import elicit_priors_with_gemini, BenchmarksResponse, print_benchmarks

from convert_benchmarks_to_priors import main as convert_benchmarks_to_priors

# Import marketing chatbot components directly instead of using system calls
from marketing_chatbot import MarketingChatbot
from simulation_agent import SimulationReACTAgent

@dataclass
class ExtractedParameters:
    """Container for extracted parameters"""
    company: str
    industry: Optional[str] = None
    geography: Optional[str] = None
    channels: Optional[List[str]] = None
    model_name: str = "gemini-2.5-flash"
    use_grounding: bool = True

class EnhancedParameterExtractor:
    """Extracts parameters from user input using Gemini API"""
    
    def __init__(self):
        self.default_channels = ["Google", "Meta", "TikTok", "LinkedIn"]
        self.extraction_prompt = self._create_extraction_prompt()
        
    def _create_extraction_prompt(self) -> str:
        """Create the parameter extraction prompt"""
        return """
You are a parameter extraction assistant for the `elicit_priors_with_gemini` function. This function generates marketing channel benchmarks (CPM, CTR, CVR, AOV) for companies using the Gemini AI API.

Please extract the following parameters from the user's request:

**REQUIRED PARAMETERS:**
1. **company** (str): The name of the company or business for which to generate marketing benchmarks
   - This is the only mandatory parameter
   - Examples: "Acme Corp", "TechStartup Inc", "Local Restaurant"

**OPTIONAL PARAMETERS:**
2. **industry** (str, optional): The industry or business model context
   - Examples: "B2B SaaS", "E-commerce", "Healthcare", "Restaurant", "Real Estate"
   - Helps provide more accurate industry-specific benchmarks

3. **geography** (str, optional): The country/region the estimates apply to
   - Examples: "United States", "Europe", "Asia-Pacific", "Canada", "United Kingdom"
   - Affects pricing and performance expectations

4. **channels** (List[str], optional): Specific marketing channels to analyze
   - Default: ["Google", "Meta", "TikTok", "LinkedIn"]
   - Other options: "YouTube", "Pinterest", "Snapchat", "Twitter/X", "Amazon Ads"
   - Can be a subset of the default list

5. **model_name** (str, optional): Gemini model to use
   - Default: "gemini-2.5-flash" (recommended for speed + grounding support)
   - Alternative: "gemini-2.0-flash-exp" (if available)

6. **use_grounding** (bool, optional): Whether to use Google Search grounding
   - Default: True (provides more up-to-date data but requires manual JSON parsing)
   - Set to False for guaranteed structured JSON output (no grounding)

**EXTRACTION INSTRUCTIONS:**
- If the user doesn't specify a parameter, use the default values
- For channels, if the user mentions specific platforms, extract those; otherwise use defaults
- For industry and geography, try to infer from context if not explicitly stated
- Always extract the company name - this is required
- Return the parameters in a structured format ready for the function call

**EXAMPLE EXTRACTION:**
User: "I need marketing benchmarks for a B2B SaaS company in the US"
Extracted:
- company: "B2B SaaS company" (or ask for specific company name)
- industry: "B2B SaaS"
- geography: "United States"
- channels: ["Google", "Meta", "TikTok", "LinkedIn"] (default)
- model_name: "gemini-2.5-flash" (default)
- use_grounding: True (default)

Please extract the parameters from the user's request now.
"""
    
    def extract_parameters_with_gemini(self, user_input: str) -> ExtractedParameters:
        """Extract parameters using Gemini API"""
        try:
            from google import genai
            
            # Check if Gemini API key is available
            if not os.getenv("GEMINI_API_KEY"):
                print("⚠️  Gemini API key not found, falling back to interactive extraction")
                return self._interactive_extraction(user_input)
            
            print("🤖 Using Gemini API for parameter extraction...")
            
            # Initialize Gemini client
            client = genai.Client()
            
            # Create the extraction prompt
            full_prompt = f"{self.extraction_prompt}\n\nUser request: {user_input}\n\nPlease extract the parameters and return them in this exact JSON format:\n{{\n  \"company\": \"company_name\",\n  \"industry\": \"industry_name or null\",\n  \"geography\": \"geography_name or null\",\n  \"channels\": [\"channel1\", \"channel2\"] or null,\n  \"model_name\": \"gemini_model_name\",\n  \"use_grounding\": true or false\n}}"
            
            # Call Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",  # Use a smaller model for extraction
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            if hasattr(response, 'parsed') and response.parsed:
                # Parse the structured response
                extracted_data = response.parsed
                return self._parse_extracted_data(extracted_data)
            else:
                # Fallback to text parsing
                print("⚠️  Structured response failed, parsing text response...")
                return self._parse_text_response(response.text, user_input)
                
        except Exception as e:
            print(f"⚠️  Gemini extraction failed: {e}")
            print("🔄 Falling back to interactive extraction...")
            return self._interactive_extraction(user_input)
    
    def _parse_extracted_data(self, data: Any) -> ExtractedParameters:
        """Parse structured data from Gemini"""
        try:
            # Handle different response formats
            if hasattr(data, 'company'):
                # Direct object access
                return ExtractedParameters(
                    company=data.company,
                    industry=getattr(data, 'industry', None),
                    geography=getattr(data, 'geography', None),
                    channels=getattr(data, 'channels', None),
                    model_name=getattr(data, 'model_name', 'gemini-2.5-flash'),
                    use_grounding=getattr(data, 'use_grounding', True)
                )
            elif isinstance(data, dict):
                # Dictionary format
                return ExtractedParameters(
                    company=data.get('company', 'Unknown Company'),
                    industry=data.get('industry'),
                    geography=data.get('geography'),
                    channels=data.get('channels'),
                    model_name=data.get('model_name', 'gemini-2.5-flash'),
                    use_grounding=data.get('use_grounding', True)
                )
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")
        except Exception as e:
            print(f"⚠️  Failed to parse structured data: {e}")
            raise
    
    def _parse_text_response(self, text: str, user_input: str) -> ExtractedParameters:
        """Parse text response from Gemini"""
        try:
            # Try to extract JSON from the response
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                print(f"Correctly parsed JSON   ")
                json_str = json_match.group(0)
                data = json.loads(json_str)
                parsed_data = self._parse_extracted_data(data)
                print(f"Parsed data for extraction success! {parsed_data}")
                return parsed_data
            else:
                # No JSON found, fall back to interactive
                print("⚠️  No JSON found in response, using interactive extraction")
                return self._interactive_extraction(user_input)
                
        except Exception as e:
            print(f"⚠️  Failed to parse text response: {e}")
            return self._interactive_extraction(user_input)
    
    def _interactive_extraction(self, user_input: str) -> ExtractedParameters:
        """Interactive parameter extraction as fallback"""
        print("🔍 Interactive Parameter Extraction")
        print("Please provide the following information:")
        
        # Company (required)
        while True:
            company = input("🏢 Company name (required): ").strip()
            if company:
                break
            print("❌ Company name is required!")
        
        # Industry
        industry = input("🏭 Industry (optional, press Enter to skip): ").strip()
        industry = industry if industry else None
        
        # Geography
        geography = input("🌍 Geography (optional, press Enter to skip): ").strip()
        geography = geography if geography else None
        
        # Channels
        print("📱 Available channels: Google, Meta, TikTok, LinkedIn, YouTube, Pinterest, Snapchat, Twitter/X, Amazon Ads")
        channels_input = input("Channels (comma-separated, or press Enter for default): ").strip()
        if channels_input:
            channels = [ch.strip() for ch in channels_input.split(",")]
        else:
            channels = self.default_channels
        
        # Model
        model_name = input(f"🤖 Gemini model (press Enter for gemini-2.5-flash): ").strip()
        model_name = model_name if model_name else "gemini-2.5-flash"
        
        # Grounding
        grounding_input = input("🔍 Use Google Search grounding? (y/n, default: y): ").strip().lower()
        use_grounding = grounding_input not in ['n', 'no', 'false']
        
        return ExtractedParameters(
            company=company,
            industry=industry,
            geography=geography,
            channels=channels,
            model_name=model_name,
            use_grounding=use_grounding
        )

class EnhancedInteractiveMarketingAgent:
    """Enhanced interactive agent class with Gemini-powered parameter extraction"""
    
    def __init__(self):
        self.parameter_extractor = EnhancedParameterExtractor()
        self.gemini_results: Optional[BenchmarksResponse] = None
    
    def run(self):
        """Main interaction loop"""
        print("🤖 Enhanced Interactive Marketing Agent")
        print("=" * 50)
        print("I'll help you generate marketing benchmarks using Gemini AI and then proceed with marketing tasks.")
        print("This version uses Gemini API for intelligent parameter extraction!")
        print()
        
        while True:
            try:
                # Get user input
                user_input = self._get_user_input()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                

                
                # Extract parameters using Gemini
                print("\n🔍 Extracting parameters using Gemini AI...")
                params = self._extract_and_confirm_parameters(user_input)
                
                # Call Gemini for benchmarks
                print("\n🚀 Calling Gemini AI for marketing benchmarks...")
                self.gemini_results = self._call_gemini(params)
                
                # Show results and get confirmation
                print("\n📊 Gemini Results:")
                print_benchmarks(self.gemini_results)
                
                if self._get_user_confirmation("Do you want to proceed with these benchmarks?"):
                    # Call marketing agent
                    print("\n🔄 Proceeding with marketing agent...")
                    self._call_marketing_agent()
                else:
                    print("🔄 Restarting parameter extraction...")
                    continue
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🔄 Let's try again...")
                continue
    
    def _get_user_input(self) -> str:
        """Get user input for parameter extraction"""
        print("\n💬 Please describe what you need:")
        print("Examples:")
        print("- 'I need marketing benchmarks for a B2B SaaS company in the US'")
        print("- 'Generate benchmarks for Acme Corp, e-commerce business'")
        print("- 'Marketing data for TechStartup Inc, healthcare industry'")
        print("- 'Benchmarks for Google and Meta ads for a restaurant in Canada'")
        print("- Type 'quit' to exit")
        print()
        
        return input("Your request: ").strip()
    
    def _extract_and_confirm_parameters(self, user_input: str) -> ExtractedParameters:
        """Extract parameters and confirm with user"""
        # Use Gemini-powered extraction
        params = self.parameter_extractor.extract_parameters_with_gemini(user_input)
        
        print("\n📋 Extracted Parameters:")
        print(f"  Company: {params.company}")
        print(f"  Industry: {params.industry or 'Not specified'}")
        print(f"  Geography: {params.geography or 'Not specified'}")
        print(f"  Channels: {params.channels or 'Default: Google, Meta, TikTok, LinkedIn'}")
        print(f"  Model: {params.model_name}")
        print(f"  Use Grounding: {params.use_grounding}")
        
        # Allow user to modify parameters
        if self._get_user_confirmation("Would you like to modify any parameters?"):
            params = self._modify_parameters(params)
        
        return params
    
    def _modify_parameters(self, params: ExtractedParameters) -> ExtractedParameters:
        """Allow user to modify extracted parameters"""
        print("\n✏️  Parameter Modification:")
        
        # Company
        new_company = input(f"Company [{params.company}]: ").strip()
        if new_company:
            params.company = new_company
        
        # Industry
        new_industry = input(f"Industry [{params.industry or 'Not specified'}]: ").strip()
        if new_industry:
            params.industry = new_industry if new_industry != "Not specified" else None
        
        # Geography
        new_geography = input(f"Geography [{params.geography or 'Not specified'}]: ").strip()
        if new_geography:
            params.geography = new_geography if new_geography != "Not specified" else None
        
        # Channels
        print(f"Current channels: {params.channels or ['Google', 'Meta', 'TikTok', 'LinkedIn']}")
        new_channels_input = input("New channels (comma-separated, or press Enter to keep current): ").strip()
        if new_channels_input:
            params.channels = [ch.strip() for ch in new_channels_input.split(",")]
        
        # Model
        new_model = input(f"Model [{params.model_name}]: ").strip()
        if new_model:
            params.model_name = new_model
        
        # Grounding
        grounding_input = input(f"Use grounding [{params.use_grounding}]: ").strip().lower()
        if grounding_input in ['true', 'yes', 'y']:
            params.use_grounding = True
        elif grounding_input in ['false', 'no', 'n']:
            params.use_grounding = False
        
        return params
    
    def _call_gemini(self, params: ExtractedParameters) -> BenchmarksResponse:
        """Call the Gemini API to get marketing benchmarks"""
        try:
            result = elicit_priors_with_gemini(
                company=params.company,
                industry=params.industry,
                geography=params.geography,
                channels=params.channels,
                model_name=params.model_name,
                use_grounding=params.use_grounding
            )
            return result
        except Exception as e:
            print(f"❌ Error calling Gemini: {e}")
            raise
    
    def _get_user_confirmation(self, message: str) -> bool:
        """Get user confirmation for a yes/no question"""
        while True:
            response = input(f"{message} (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _call_marketing_agent(self):
        """Call the marketing agent for remaining tasks directly instead of using system calls"""
        print("\n🎯 Launching Marketing Agent...")
        
        try:
            # Save Gemini results to a temporary file for the marketing agent
            temp_file = f"marketing_benchmarks_config.json"
            self._save_gemini_results(temp_file)
            
            print(f"💾 Gemini results saved to: {temp_file}")
            print("🔄 Converting benchmarks to channel priors...")
            
            # Convert the gemini results to channel priors
            convert_benchmarks_to_priors()
            
            print("🚀 Starting marketing optimization session...")
            
            # Check if OpenAI API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ Error: OpenAI API key not found!")
                print("💡 Please set the OPENAI_API_KEY environment variable to use the marketing agent.")
                print("Example: export OPENAI_API_KEY='your-api-key-here'")
                return
            
            # Create and run the marketing chatbot directly
            chatbot = MarketingChatbot(api_key=api_key)
            
            # Option 1: Run interactive session (default)
            chatbot.run_interactive_session()
            
            # Option 2: For programmatic use, you can also use:
            # response = chatbot.process_single_query("Optimize $50k budget for maximum revenue")
            # print(response)
            
        except Exception as e:
            print(f"❌ Error launching marketing agent: {e}")
            print("💡 You can manually run: python marketing_chatbot.py")
    
    def _save_gemini_results(self, filename: str):
        """Save Gemini results to a JSON file"""
        if self.gemini_results:
            # Convert to dict and save with correct structure
            data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "tool": "enhanced_interactive_agent",
                    "model": "gemini-2.5-flash"
                },
                "benchmarks": {
                    "company": self.gemini_results.company,
                    "industry": self.gemini_results.industry,
                    "geography": self.gemini_results.geography,
                    "timeframe": self.gemini_results.timeframe,
                    "channels": []
                }
            }
            
            for channel in self.gemini_results.channels:
                channel_data = {
                    "channel": channel.channel,
                    "geography": channel.geography,
                    "industry": channel.industry,
                    "cpm_usd_per_1000": {
                        "unit": channel.cpm_usd_per_1000.unit,
                        "p10": channel.cpm_usd_per_1000.p10,
                        "p50": channel.cpm_usd_per_1000.p50,
                        "p90": channel.cpm_usd_per_1000.p90
                    },
                    "ctr_ratio": {
                        "unit": channel.ctr_ratio.unit,
                        "p10": channel.ctr_ratio.p10,
                        "p50": channel.ctr_ratio.p50,
                        "p90": channel.ctr_ratio.p90
                    },
                    "cvr_ratio": {
                        "unit": channel.cvr_ratio.unit,
                        "p10": channel.cvr_ratio.p10,
                        "p50": channel.cvr_ratio.p50,
                        "p90": channel.cvr_ratio.p90
                    },
                    "aov_usd": {
                        "unit": channel.aov_usd.unit,
                        "p10": channel.aov_usd.p10,
                        "p50": channel.aov_usd.p50,
                        "p90": channel.aov_usd.p90
                    },
                    "notes": channel.notes,
                    "sources": [{"title": s.title, "uri": s.uri} for s in (channel.sources or [])]
                }
                data["benchmarks"]["channels"].append(channel_data)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to: {filename}")

def main():
    """Main entry point"""
    # Check if Gemini API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("💡 Please set it with: export GEMINI_API_KEY='your_key'")
        return
    
    # Check if required files exist
    if not os.path.exists("elict_priors_gemini.py"):
        print("❌ Error: elict_priors_gemini.py not found")
        print("💡 Please ensure this file is in the same directory")
        return
    
    # Launch enhanced interactive agent
    agent = EnhancedInteractiveMarketingAgent()
    agent.run()

if __name__ == "__main__":
    main() 