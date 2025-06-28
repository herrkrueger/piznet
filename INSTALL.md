# 📦 Installation Guide - Patent Intelligence Platform

Complete installation guide for setting up the Patent Intelligence Platform on your local machine or server.

## 🎯 Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/piznet.git
cd piznet

# 2. Run automated setup
python setup.py

# 3. Configure credentials (edit the .env file)
cp .env.template .env
# Edit .env with your actual API credentials

# 4. Run demo
jupyter notebook notebooks/Patent_Intelligence_Platform_Demo.ipynb
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for cache and exports
- **Internet**: Required for EPO OPS API and PATSTAT access

### Required Credentials
- **EPO OPS API**: Consumer key and secret ([Register here](https://ops.epo.org))
- **PATSTAT Access**: Username/password (EPO TIP environment preferred)
- **Google Cloud** (Optional): Service account for external BigQuery access

## 🔧 Installation Methods

### Method 1: Automated Setup (Recommended)

The automated setup script handles everything for you:

```bash
python setup.py
```

**What it does:**
- ✅ Checks Python version compatibility
- ✅ Detects EPO TIP environment
- ✅ Installs all dependencies
- ✅ Sets up directory structure
- ✅ Creates .env configuration file
- ✅ Validates installation

### Method 2: Manual Installation

#### Step 1: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# For development:
pip install -r requirements-dev.txt
```

#### Step 2: Setup Directories
```bash
mkdir -p logs exports cache/analysis cache/epo_ops cache/patstat
```

#### Step 3: Configure Environment
```bash
cp .env.template .env
# Edit .env file with your credentials
```

## 🔐 Configuration Setup

### Environment Variables (.env file)

Copy `.env.template` to `.env` and configure:

```bash
# EPO OPS API Credentials (Required)
OPS_KEY=your_epo_ops_consumer_key
OPS_SECRET=your_epo_ops_consumer_secret

# PATSTAT Database (Required for full functionality)
PATSTAT_USER=your_patstat_username
PATSTAT_PASSWORD=your_patstat_password

# Optional: Google Cloud BigQuery
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Application Settings
ENVIRONMENT=development
CACHE_DIRECTORY=./cache
LOG_LEVEL=INFO
```

### Getting EPO OPS Credentials

1. Visit [EPO OPS Registration](https://ops.epo.org)
2. Create a developer account
3. Register your application
4. Copy Consumer Key and Consumer Secret to .env file

### PATSTAT Access

**Option A: EPO TIP Environment (Recommended)**
- Direct access to production PATSTAT database
- Requires EPO TIP environment access
- Best performance and data coverage

**Option B: External BigQuery**
- Access PATSTAT via Google Cloud BigQuery
- Requires Google Cloud service account
- Good for external deployments

## 🧪 Testing Your Installation

### Quick Validation
```bash
# Test configuration
python config/test_config.py

# Test data access
python data_access/test_data_access.py

# Test processors
python processors/test_unit.py
```

### Shell Script Tests
```bash
# All-in-one tests
./test_config.sh
./test_data_access.sh  
./test_processors.sh
```

### Full Demo
```bash
# Run complete platform demo
jupyter notebook notebooks/Patent_Intelligence_Platform_Demo.ipynb
```

## 📁 Directory Structure

After installation, your directory structure will be:

```
piznet/
├── 📁 config/              # YAML configuration files
├── 📁 data_access/         # Database and API clients
├── 📁 processors/          # Analysis processors
├── 📁 visualizations/      # Charts, dashboards, maps
├── 📁 analyzers/           # Advanced analytics modules
├── 📁 notebooks/           # Jupyter demo notebooks
├── 📁 scripts/             # Utility and test scripts
├── 📁 cache/               # Cached data (auto-created)
│   ├── analysis/
│   ├── epo_ops/
│   └── patstat/
├── 📁 logs/                # Application logs (auto-created)
├── 📁 exports/             # Generated reports (auto-created)
├── 📄 .env                 # Your configuration (create from template)
├── 📄 requirements.txt     # Production dependencies
├── 📄 requirements-dev.txt # Development dependencies
├── 📄 Pipfile              # Pipenv configuration
└── 📄 setup.py             # Automated setup script
```

## 🚀 Running the Platform

### Development Mode
```bash
# Run Jupyter notebook
jupyter notebook

# Open: notebooks/Patent_Intelligence_Platform_Demo.ipynb
```

### Production Mode
```bash
# Set production environment
export ENVIRONMENT=production

# Run analysis script
python scripts/test_platform_workflow.py
```

### Custom Analysis
```python
from config import ConfigurationManager
from data_access import PatstatClient, PatentSearcher
from processors import ApplicantAnalyzer
from visualizations import create_executive_analysis

# Initialize platform
config = ConfigurationManager()
patstat = PatstatClient(environment='PROD')
searcher = PatentSearcher(patstat)

# Execute search
results = searcher.execute_comprehensive_search(
    start_date='2024-01-01', 
    end_date='2024-01-07'
)

# Create analysis
analysis = create_executive_analysis(results)
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**PATSTAT connection failed**
```bash
# Check .env file configuration
cat .env

# Test configuration
python config/test_config.py
```

**EPO OPS authentication failed**
```bash
# Verify credentials in .env
# Check EPO OPS registration status
# Test with: python data_access/test_data_access.py
```

**Permission denied errors**
```bash
# Check directory permissions
chmod 755 logs/ exports/ cache/

# On Windows, run as Administrator if needed
```

### Performance Issues

**Slow search queries**
- Check PATSTAT connection settings
- Adjust batch sizes in configuration
- Enable caching in api_config.yaml

**High memory usage**
- Reduce search date ranges
- Clear cache: `rm -rf cache/*`
- Monitor with: `python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"`

### Getting Help

1. **Configuration Issues**: Run `python config/test_config.py`
2. **Data Access Problems**: Run `python data_access/test_data_access.py`
3. **Platform Issues**: Check `logs/` directory for error details
4. **GitHub Issues**: Report bugs at repository issues page

## 🔄 Updating the Platform

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Run tests to verify
./test_config.sh
```

## 🏗️ Development Setup

For contributing to the platform:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black .              # Format code
flake8 .            # Lint code
mypy .              # Type checking
pytest              # Run tests
```

## 📊 Performance Benchmarks

**Typical Performance** (PROD environment):
- Search execution: 281 patents in ~30 seconds
- Analysis pipeline: 4 processors in ~45 seconds
- Visualization generation: 6 charts in ~15 seconds
- Export creation: 6 files in ~5 seconds

**System Requirements** for optimal performance:
- **Small datasets** (<1k patents): 4GB RAM, 2 CPU cores
- **Medium datasets** (1k-10k patents): 8GB RAM, 4 CPU cores
- **Large datasets** (>10k patents): 16GB RAM, 8 CPU cores

---

## 🎉 You're Ready!

Once installation is complete, you'll have a production-ready patent intelligence platform. Start with the demo notebook and explore the comprehensive analytics capabilities!

**Next Steps:**
1. ✅ Complete installation
2. 🔧 Configure credentials  
3. 🧪 Run tests
4. 🚀 Launch demo
5. 📊 Explore analytics