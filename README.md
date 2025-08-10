# Setup Guide for Enhanced Interactive Marketing Agent

This guide will help you set up and run the `enhanced_interactive_agent2.py` file, which is an AI-powered marketing budget optimization tool.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Step-by-Step Setup

### 1. Clone/Setup the Repository
If you haven't already, navigate to your project directory


### 2. Create a Virtual Environment (Recommended)
```bash
# Create a virtual environment
conda create --name hack-leo python=3.10

# Activate the conda environment
conda activate hack-leo
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Set Up API Keys

You'll need two API keys for full functionality:

#### Gemini API Key (Required)
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Or add to your shell profile (~/.zshrc, ~/.bashrc, etc.)
echo 'export GEMINI_API_KEY="your_gemini_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

**To get a Gemini API key:**
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API key" in the top right
4. Copy the API key

#### OpenAI API Key (Required for Marketing Agent)
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Or add to your shell profile
echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

**To get an OpenAI API key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Go to API Keys section
4. Create a new API key

### 5. Verify File Structure
Ensure these files exist in your project directory:
- ✅ `enhanced_interactive_agent2.py` (main file)
- ✅ `elict_priors_gemini.py` (Gemini integration)
- ✅ `convert_benchmarks_to_priors.py` (data conversion for ranges given by gemini to mu/sigma)
- ✅ `marketing_chatbot.py` (chatbot interface)
- ✅ `simulation_agent.py` (simulation logic)
- ✅ `simulation_models.py` (data models)
- ✅ `simulation_tools.py` (simulation tools)
- ✅ `modelling_adv_goals2.py` (core simulation engine)

### 6. Run the Application
```bash
# Make sure your virtual environment is activated

# Run the enhanced interactive agent
python enhanced_interactive_agent2.py
```

## How It Works

1. **Parameter Extraction**: The agent uses Gemini AI to intelligently extract marketing parameters from your natural language input
2. **Benchmark Generation**: Gemini generates marketing channel benchmarks (CPM, CTR, CVR, AOV) for your company
3. **Budget Optimization**: The marketing agent runs Monte Carlo simulations to find optimal budget allocations
4. **Visualization**: Results are displayed with charts and analysis

## Example Usage

When you run the application, you can ask questions like:
- "I need marketing benchmarks for a B2B SaaS company in the US"
- "Generate benchmarks for Acme Corp, e-commerce business"
- "Marketing data for TechStartup Inc, healthcare industry"

Then when the marketing agent pops up let it know your budget and goal i.e revenue, profit, conversions etc (can easily add more objectives in future)

### File Dependencies

The main file imports these modules:
- `elict_priors_gemini.py` - Gemini API integration for marketing benchmarks
- `convert_benchmarks_to_priors.py` - Converts benchmarks to simulation parameters
- `marketing_chatbot.py` - Main chatbot interface
- `simulation_agent.py` - ReACT agent for simulation requests
- `simulation_models.py` - Pydantic models for data validation
- `simulation_tools.py` - Tools for running simulations
- `modelling_adv_goals2.py` - Core simulation engine

## Features

- 🤖 **AI-Powered Parameter Extraction** using Gemini
- 📊 **Marketing Benchmark Generation** for various industries and channels
- 💰 **Budget Optimization** with Monte Carlo simulations
- 📈 **Data Visualization** with charts and analysis
- 🔄 **Interactive Workflow** with confirmation steps
- 🌍 **Multi-Channel Support** (Google, Meta, TikTok, LinkedIn, etc.)

## Next Steps

After successful setup:
1. Try running the application with a simple query
2. Experiment with different company types and industries
3. Explore the generated visualizations in the `plots/` directory
4. Use the marketing agent for budget optimization questions

