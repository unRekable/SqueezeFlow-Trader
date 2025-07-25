#!/usr/bin/env python3
"""
SqueezeFlow Trader - System Status Checker
Quick status check for all system components
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    import redis
    from influxdb import InfluxDBClient
except ImportError:
    print("⚠️  Some dependencies missing. Run: pip install -r requirements.txt")
    sys.exit(1)


class SystemStatusChecker:
    """Check status of all SqueezeFlow Trader components"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.status_results = {}
        
    def check_all_services(self) -> Dict[str, bool]:
        """Check status of all services"""
        print("🔍 SqueezeFlow Trader - System Status Check")
        print("=" * 50)
        
        checks = [
            ("Docker Services", self._check_docker_services),
            ("InfluxDB", self._check_influxdb),
            ("Redis", self._check_redis), 
            ("aggr-server", self._check_aggr_server),
            ("Freqtrade", self._check_freqtrade),
            ("Grafana", self._check_grafana),
            ("System Files", self._check_system_files),
        ]
        
        all_healthy = True
        
        for service_name, check_func in checks:
            print(f"\n📊 {service_name}:")
            try:
                is_healthy = check_func()
                self.status_results[service_name] = is_healthy
                
                if is_healthy:
                    print(f"  ✅ {service_name} - HEALTHY")
                else:
                    print(f"  ❌ {service_name} - UNHEALTHY")
                    all_healthy = False
                    
            except Exception as e:
                print(f"  ❌ {service_name} - ERROR: {e}")
                self.status_results[service_name] = False
                all_healthy = False
        
        # Print summary
        print("\n" + "=" * 50)
        if all_healthy:
            print("✅ ALL SERVICES HEALTHY")
            print("🚀 System is operational")
        else:
            print("⚠️  SOME SERVICES HAVE ISSUES")
            self._print_troubleshooting_tips()
            
        return self.status_results
    
    def _check_docker_services(self) -> bool:
        """Check Docker services status"""
        try:
            # Check if Docker is running
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                check=True
            )
            print("  ✓ Docker daemon running")
            
            # Check docker-compose services
            result = subprocess.run(
                ["docker-compose", "ps"], 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                
                # Parse service status
                running_services = []
                stopped_services = []
                
                for line in output_lines[2:]:  # Skip header lines
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            service_name = parts[0]
                            if "Up" in line:
                                running_services.append(service_name)
                                print(f"  ✓ {service_name} - Running")
                            else:
                                stopped_services.append(service_name)
                                print(f"  ❌ {service_name} - Stopped")
                
                return len(stopped_services) == 0
            else:
                print("  ❌ Docker Compose not running")
                return False
                
        except subprocess.CalledProcessError:
            print("  ❌ Docker not running or not accessible")
            return False
        except FileNotFoundError:
            print("  ❌ Docker not installed")
            return False
    
    def _check_influxdb(self) -> bool:
        """Check InfluxDB connectivity"""
        try:
            client = InfluxDBClient(
                host='localhost',
                port=8086,
                username='squeezeflow',
                password='password123',
                database='significant_trades'
            )
            
            # Test connection
            databases = client.get_list_database()
            print(f"  ✓ Connected to InfluxDB")
            print(f"  ✓ Found {len(databases)} databases")
            
            # Check if our database exists
            db_exists = any(db['name'] == 'significant_trades' for db in databases)
            if db_exists:
                print("  ✓ significant_trades database exists")
            else:
                print("  ⚠️  significant_trades database missing")
                
            return True
            
        except Exception as e:
            print(f"  ❌ InfluxDB connection failed: {e}")
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis connectivity"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            
            # Test connection
            r.ping()
            print("  ✓ Redis connection successful")
            
            # Check memory usage
            info = r.info()
            memory_used = info.get('used_memory_human', 'Unknown')
            print(f"  ✓ Memory usage: {memory_used}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Redis connection failed: {e}")
            return False
    
    def _check_aggr_server(self) -> bool:
        """Check aggr-server status"""
        try:
            response = requests.get('http://localhost:3000', timeout=5)
            
            if response.status_code == 200:
                print("  ✓ aggr-server responding")
                return True
            else:
                print(f"  ❌ aggr-server returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("  ❌ aggr-server not accessible")
            return False
        except Exception as e:
            print(f"  ❌ aggr-server check failed: {e}")
            return False
    
    def _check_freqtrade(self) -> bool:
        """Check Freqtrade API status"""
        try:
            # Check Freqtrade API
            response = requests.get('http://localhost:8080/api/v1/ping', timeout=5)
            
            if response.status_code == 200:
                print("  ✓ Freqtrade API responding")
                
                # Check if trading is active
                status_response = requests.get('http://localhost:8080/api/v1/status', timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get('runmode'):
                        print(f"  ✓ Freqtrade mode: {status_data['runmode']}")
                    
                return True
            else:
                print(f"  ❌ Freqtrade API returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("  ❌ Freqtrade API not accessible")
            return False
        except Exception as e:
            print(f"  ❌ Freqtrade check failed: {e}")
            return False
    
    def _check_grafana(self) -> bool:
        """Check Grafana status"""
        try:
            response = requests.get('http://localhost:3002/login', timeout=5)
            
            if response.status_code == 200:
                print("  ✓ Grafana accessible")
                return True
            else:
                print(f"  ❌ Grafana returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("  ❌ Grafana not accessible")
            return False
        except Exception as e:
            print(f"  ❌ Grafana check failed: {e}")
            return False
    
    def _check_system_files(self) -> bool:
        """Check critical system files"""
        critical_files = [
            "main.py",
            "requirements.txt",
            "docker-compose.yml",
            ".env",
            "config/config.yaml"
        ]
        
        missing_files = []
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"  ✓ {file_path}")
            else:
                print(f"  ❌ {file_path} - MISSING")
                missing_files.append(file_path)
        
        return len(missing_files) == 0
    
    def _print_troubleshooting_tips(self):
        """Print troubleshooting tips"""
        print("\n🔧 Troubleshooting Tips:")
        print("-" * 30)
        
        for service, is_healthy in self.status_results.items():
            if not is_healthy:
                if service == "Docker Services":
                    print("• Start Docker services: ./start.sh")
                    print("• Check Docker logs: docker-compose logs")
                elif service == "InfluxDB":
                    print("• Wait for InfluxDB startup (may take 30-60 seconds)")
                    print("• Check logs: docker-compose logs aggr-influx")
                elif service == "Redis":
                    print("• Restart Redis: docker-compose restart redis")
                elif service == "aggr-server":
                    print("• Check aggr-server logs: docker-compose logs aggr-server")
                elif service == "Freqtrade":
                    print("• Start Freqtrade: docker-compose up -d freqtrade")
                elif service == "Grafana":
                    print("• Start Grafana: docker-compose up -d grafana")
                elif service == "System Files":
                    print("• Run initialization: python init.py --force")


def main():
    """Main status check function"""
    checker = SystemStatusChecker()
    results = checker.check_all_services()
    
    # Exit with error code if any service is unhealthy
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()