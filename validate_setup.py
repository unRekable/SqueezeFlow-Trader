#!/usr/bin/env python3
"""
SqueezeFlow Trader - Setup Validation Script
Validates that the system is properly configured and ready to run
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


class SetupValidator:
    """Validate SqueezeFlow Trader setup"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.validation_results = []
        
    def run_validation(self) -> bool:
        """Run complete setup validation"""
        print("🔍 Validating SqueezeFlow Trader Setup")
        print("=" * 45)
        
        # Run all validation checks
        checks = [
            ("Python Environment", self._check_python_environment),
            ("Directory Structure", self._check_directory_structure),  
            ("Configuration Files", self._check_configuration_files),
            ("Environment Variables", self._check_environment_variables),
            ("Dependencies", self._check_dependencies),
            ("Docker Setup", self._check_docker_setup),
            ("System Resources", self._check_system_resources),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}:")
            try:
                result = check_func()
                if result:
                    print(f"  ✅ {check_name} - PASSED")
                else:
                    print(f"  ❌ {check_name} - FAILED")
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {check_name} - ERROR: {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 45)
        if all_passed:
            print("✅ All validation checks PASSED!")
            print("🚀 System is ready to run")
        else:
            print("❌ Some validation checks FAILED!")
            print("🔧 Please fix the issues before running the system")
            
        return all_passed
    
    def _check_python_environment(self) -> bool:
        """Check Python environment"""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                print("  ❌ Python 3.8+ required")
                return False
            print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
            
            # Check virtual environment
            venv_path = self.project_root / ".venv"
            if venv_path.exists():
                print("  ✓ Virtual environment found")
            else:
                print("  ⚠️  Virtual environment not found")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error checking Python: {e}")
            return False
    
    def _check_directory_structure(self) -> bool:
        """Check required directory structure"""
        required_dirs = [
            "config",
            "data",
            "data/logs", 
            "freqtrade/user_data",
            "state",
            "models"
        ]
        
        missing_dirs = []
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ❌ {dir_path} - MISSING")
                missing_dirs.append(dir_path)
        
        return len(missing_dirs) == 0
    
    def _check_configuration_files(self) -> bool:
        """Check configuration files"""
        required_configs = [
            "config.yaml",
            "exchanges.yaml", 
            "risk_management.yaml",
            "trading_parameters.yaml"
        ]
        
        config_dir = self.project_root / "config"
        missing_configs = []
        
        for config_file in required_configs:
            config_path = config_dir / config_file
            if config_path.exists():
                try:
                    # Try to parse YAML
                    with open(config_path, 'r') as f:
                        yaml.safe_load(f)
                    print(f"  ✓ {config_file}")
                except yaml.YAMLError as e:
                    print(f"  ❌ {config_file} - INVALID YAML: {e}")
                    return False
            else:
                print(f"  ❌ {config_file} - MISSING")
                missing_configs.append(config_file)
        
        return len(missing_configs) == 0
    
    def _check_environment_variables(self) -> bool:
        """Check environment variables"""
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            print("  ❌ .env file missing")
            return False
        
        print("  ✓ .env file exists")
        
        # Check for required variables
        required_vars = [
            "INFLUX_HOST",
            "INFLUX_PORT", 
            "REDIS_URL"
        ]
        
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            for var in required_vars:
                if f"{var}=" in env_content:
                    print(f"  ✓ {var}")
                else:
                    print(f"  ⚠️  {var} not set")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error reading .env: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check Python dependencies"""
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("  ❌ requirements.txt missing")
            return False
        
        print("  ✓ requirements.txt exists")
        
        # Try to import key dependencies
        key_deps = [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("yaml", "pyyaml"),
            ("redis", "redis"),
            ("requests", "requests")
        ]
        
        missing_deps = []
        
        for module_name, package_name in key_deps:
            try:
                __import__(module_name)
                print(f"  ✓ {package_name}")
            except ImportError:
                print(f"  ❌ {package_name} - NOT INSTALLED")
                missing_deps.append(package_name)
        
        return len(missing_deps) == 0
    
    def _check_docker_setup(self) -> bool:
        """Check Docker setup (optional)"""
        try:
            # Check Docker
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✓ Docker: {result.stdout.strip()}")
            else:
                print("  ⚠️  Docker not available")
                return True  # Optional for development
            
            # Check docker-compose
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✓ Docker Compose: {result.stdout.strip()}")
            else:
                print("  ⚠️  Docker Compose not available")
            
            # Check docker-compose.yml
            compose_file = self.project_root / "docker-compose.yml"
            if compose_file.exists():
                print("  ✓ docker-compose.yml exists")
            else:
                print("  ❌ docker-compose.yml missing")
                return False
            
            return True
            
        except FileNotFoundError:
            print("  ⚠️  Docker not installed (OK for development)")
            return True
        except Exception as e:
            print(f"  ❌ Docker check error: {e}")
            return False
    
    def _check_system_resources(self) -> bool:
        """Check system resources"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 4:
                print(f"  ✓ Memory: {memory_gb:.1f} GB")
            else:
                print(f"  ⚠️  Memory: {memory_gb:.1f} GB (4GB+ recommended)")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            if free_gb >= 10:
                print(f"  ✓ Disk space: {free_gb:.1f} GB free")
            else:
                print(f"  ⚠️  Disk space: {free_gb:.1f} GB free (10GB+ recommended)")
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            print(f"  ✓ CPU cores: {cpu_count}")
            
            return True
            
        except ImportError:
            print("  ⚠️  psutil not available for resource check")
            return True
        except Exception as e:
            print(f"  ❌ Resource check error: {e}")
            return False


def main():
    """Main validation function"""
    validator = SetupValidator()
    success = validator.run_validation()
    
    if not success:
        print("\n🔧 Suggested fixes:")
        print("1. Run: python init.py --mode development")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Check configuration files in config/ directory")
        sys.exit(1)
    else:
        print("\n🎯 You can now start the system:")
        print("- Development: python main.py start --dry-run")
        print("- Production: ./start.sh")


if __name__ == "__main__":
    main()