#!/usr/bin/env python3
"""
Automated Dependency Update System

Manages automatic updates of project dependencies including security patches,
version updates, and compatibility testing.
"""

import os
import json
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import requests
import semver
from packaging import version
from packaging.requirements import Requirement
import toml
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """Dependency update strategies"""
    PATCH = "patch"          # Only patch updates (1.2.3 -> 1.2.4)
    MINOR = "minor"          # Minor updates (1.2.3 -> 1.3.0)
    MAJOR = "major"          # Major updates (1.2.3 -> 2.0.0)
    SECURITY = "security"    # Only security updates
    STABLE = "stable"        # Only stable releases (no pre-releases)


class UpdatePriority(Enum):
    """Update priority levels"""
    CRITICAL = "critical"    # Security vulnerabilities
    HIGH = "high"           # Important bug fixes
    MEDIUM = "medium"       # Feature updates
    LOW = "low"            # Optional updates


@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    current_version: str
    latest_version: Optional[str] = None
    update_available: bool = False
    update_type: Optional[str] = None  # patch, minor, major
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    changelog_url: Optional[str] = None
    homepage_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    license: Optional[str] = None
    
    def get_update_priority(self) -> UpdatePriority:
        """Determine update priority"""
        if self.security_issues:
            return UpdatePriority.CRITICAL
        elif self.update_type == "major":
            return UpdatePriority.MEDIUM
        elif self.update_type == "minor":
            return UpdatePriority.HIGH
        else:
            return UpdatePriority.LOW


@dataclass
class UpdateResult:
    """Result of a dependency update"""
    dependency: str
    old_version: str
    new_version: str
    success: bool
    error_message: Optional[str] = None
    tests_passed: bool = False
    rollback_performed: bool = False
    update_time: datetime = field(default_factory=datetime.utcnow)


class DependencyUpdater:
    """Automated dependency update system"""
    
    def __init__(self, 
                 project_root: str = ".",
                 config_path: str = "configs/dependency_update_config.yaml"):
        """
        Initialize dependency updater
        
        Args:
            project_root: Root directory of the project
            config_path: Path to configuration file
        """
        self.project_root = Path(project_root)
        self.config = self._load_config(config_path)
        self.update_history: List[UpdateResult] = []
        self.vulnerability_db = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = self.project_root / config_path
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                'update_strategy': 'minor',
                'auto_merge': False,
                'test_command': 'python -m pytest',
                'security_check': True,
                'backup_enabled': True,
                'excluded_packages': [],
                'pinned_packages': {}
            }
    
    def scan_dependencies(self) -> Dict[str, List[DependencyInfo]]:
        """Scan all project dependencies"""
        dependencies = {
            'python': [],
            'npm': [],
            'docker': []
        }
        
        # Scan Python dependencies
        if (self.project_root / "requirements.txt").exists():
            dependencies['python'].extend(self._scan_requirements_txt())
        
        if (self.project_root / "pyproject.toml").exists():
            dependencies['python'].extend(self._scan_pyproject_toml())
        
        if (self.project_root / "environment.yml").exists():
            dependencies['python'].extend(self._scan_conda_env())
        
        # Scan NPM dependencies
        if (self.project_root / "package.json").exists():
            dependencies['npm'].extend(self._scan_package_json())
        
        # Scan Docker base images
        dependencies['docker'].extend(self._scan_dockerfiles())
        
        return dependencies
    
    def _scan_requirements_txt(self) -> List[DependencyInfo]:
        """Scan requirements.txt file"""
        deps = []
        req_file = self.project_root / "requirements.txt"
        
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        req = Requirement(line)
                        # Get current version from specifier
                        current_version = self._extract_version_from_requirement(req)
                        
                        dep = DependencyInfo(
                            name=req.name,
                            current_version=current_version or "unknown"
                        )
                        
                        # Check for updates
                        self._check_pypi_updates(dep)
                        deps.append(dep)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing requirement '{line}': {str(e)}")
        
        return deps
    
    def _scan_pyproject_toml(self) -> List[DependencyInfo]:
        """Scan pyproject.toml file"""
        deps = []
        pyproject_file = self.project_root / "pyproject.toml"
        
        with open(pyproject_file, 'r') as f:
            data = toml.load(f)
        
        # Check dependencies section
        dependencies = data.get('project', {}).get('dependencies', [])
        for dep_str in dependencies:
            try:
                req = Requirement(dep_str)
                current_version = self._extract_version_from_requirement(req)
                
                dep = DependencyInfo(
                    name=req.name,
                    current_version=current_version or "unknown"
                )
                
                self._check_pypi_updates(dep)
                deps.append(dep)
                
            except Exception as e:
                logger.warning(f"Error parsing dependency '{dep_str}': {str(e)}")
        
        return deps
    
    def _scan_conda_env(self) -> List[DependencyInfo]:
        """Scan conda environment.yml file"""
        deps = []
        env_file = self.project_root / "environment.yml"
        
        with open(env_file, 'r') as f:
            data = yaml.safe_load(f)
        
        dependencies = data.get('dependencies', [])
        for dep in dependencies:
            if isinstance(dep, str):
                # Parse conda dependency format
                match = re.match(r'^([^=<>]+)([=<>]+.+)?$', dep)
                if match:
                    name = match.group(1)
                    version_spec = match.group(2) or ""
                    
                    dep_info = DependencyInfo(
                        name=name,
                        current_version=version_spec.lstrip('=') if version_spec else "unknown"
                    )
                    
                    # Check for updates (conda packages)
                    self._check_conda_updates(dep_info)
                    deps.append(dep_info)
        
        return deps
    
    def _scan_package_json(self) -> List[DependencyInfo]:
        """Scan package.json file"""
        deps = []
        package_file = self.project_root / "package.json"
        
        with open(package_file, 'r') as f:
            data = json.load(f)
        
        # Scan dependencies and devDependencies
        for dep_type in ['dependencies', 'devDependencies']:
            dependencies = data.get(dep_type, {})
            for name, version_spec in dependencies.items():
                dep = DependencyInfo(
                    name=name,
                    current_version=version_spec.lstrip('^~')
                )
                
                # Check NPM registry for updates
                self._check_npm_updates(dep)
                deps.append(dep)
        
        return deps
    
    def _scan_dockerfiles(self) -> List[DependencyInfo]:
        """Scan Dockerfiles for base images"""
        deps = []
        
        # Find all Dockerfiles
        dockerfiles = list(self.project_root.glob("**/Dockerfile*"))
        
        for dockerfile in dockerfiles:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            # Extract FROM statements
            from_pattern = re.compile(r'^FROM\s+([^\s]+)', re.MULTILINE)
            matches = from_pattern.findall(content)
            
            for image in matches:
                # Parse image:tag format
                if ':' in image:
                    name, tag = image.rsplit(':', 1)
                else:
                    name, tag = image, 'latest'
                
                dep = DependencyInfo(
                    name=name,
                    current_version=tag
                )
                
                # Check Docker Hub for updates
                self._check_docker_updates(dep)
                deps.append(dep)
        
        return deps
    
    def _extract_version_from_requirement(self, req: Requirement) -> Optional[str]:
        """Extract version from requirement specifier"""
        if req.specifier:
            # Get the most specific version
            for spec in req.specifier:
                if spec.operator == '==':
                    return str(spec.version)
            # If no exact version, return the first one
            if req.specifier:
                return str(list(req.specifier)[0].version)
        return None
    
    def _check_pypi_updates(self, dep: DependencyInfo) -> None:
        """Check PyPI for available updates"""
        try:
            response = requests.get(f"https://pypi.org/pypi/{dep.name}/json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                latest_version = data['info']['version']
                
                dep.latest_version = latest_version
                dep.homepage_url = data['info'].get('home_page')
                dep.license = data['info'].get('license')
                
                # Compare versions
                if dep.current_version != "unknown":
                    try:
                        current = version.parse(dep.current_version)
                        latest = version.parse(latest_version)
                        
                        if latest > current:
                            dep.update_available = True
                            dep.update_type = self._determine_update_type(
                                dep.current_version, latest_version
                            )
                    except:
                        pass
                
                # Check for security vulnerabilities
                self._check_security_vulnerabilities(dep)
                
        except Exception as e:
            logger.debug(f"Error checking PyPI for {dep.name}: {str(e)}")
    
    def _check_conda_updates(self, dep: DependencyInfo) -> None:
        """Check conda-forge for available updates"""
        try:
            # Use conda search command
            result = subprocess.run(
                ['conda', 'search', '-c', 'conda-forge', dep.name, '--json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if dep.name in data:
                    versions = [pkg['version'] for pkg in data[dep.name]]
                    if versions:
                        latest_version = max(versions, key=lambda v: version.parse(v))
                        dep.latest_version = latest_version
                        
                        if dep.current_version != "unknown":
                            try:
                                current = version.parse(dep.current_version)
                                latest = version.parse(latest_version)
                                
                                if latest > current:
                                    dep.update_available = True
                                    dep.update_type = self._determine_update_type(
                                        dep.current_version, latest_version
                                    )
                            except:
                                pass
                                
        except Exception as e:
            logger.debug(f"Error checking conda for {dep.name}: {str(e)}")
    
    def _check_npm_updates(self, dep: DependencyInfo) -> None:
        """Check NPM registry for available updates"""
        try:
            response = requests.get(f"https://registry.npmjs.org/{dep.name}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get('dist-tags', {}).get('latest')
                
                if latest_version:
                    dep.latest_version = latest_version
                    dep.homepage_url = data.get('homepage')
                    dep.license = data.get('license')
                    
                    if dep.current_version != "unknown":
                        try:
                            current = version.parse(dep.current_version)
                            latest = version.parse(latest_version)
                            
                            if latest > current:
                                dep.update_available = True
                                dep.update_type = self._determine_update_type(
                                    dep.current_version, latest_version
                                )
                        except:
                            pass
                            
        except Exception as e:
            logger.debug(f"Error checking NPM for {dep.name}: {str(e)}")
    
    def _check_docker_updates(self, dep: DependencyInfo) -> None:
        """Check Docker Hub for available updates"""
        try:
            # Check official Docker Hub API
            if '/' not in dep.name:
                # Official image
                url = f"https://hub.docker.com/v2/repositories/library/{dep.name}/tags"
            else:
                # User image
                url = f"https://hub.docker.com/v2/repositories/{dep.name}/tags"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                tags = [tag['name'] for tag in data.get('results', [])]
                
                # Find latest stable version
                version_tags = [tag for tag in tags if re.match(r'^\d+\.\d+', tag)]
                if version_tags:
                    latest_version = max(version_tags, key=lambda v: version.parse(v.split('-')[0]))
                    dep.latest_version = latest_version
                    
                    if dep.current_version not in ['latest', 'unknown']:
                        try:
                            current = version.parse(dep.current_version.split('-')[0])
                            latest = version.parse(latest_version.split('-')[0])
                            
                            if latest > current:
                                dep.update_available = True
                                dep.update_type = self._determine_update_type(
                                    dep.current_version, latest_version
                                )
                        except:
                            pass
                            
        except Exception as e:
            logger.debug(f"Error checking Docker Hub for {dep.name}: {str(e)}")
    
    def _determine_update_type(self, current: str, latest: str) -> str:
        """Determine type of update (patch, minor, major)"""
        try:
            curr_ver = semver.VersionInfo.parse(current)
            latest_ver = semver.VersionInfo.parse(latest)
            
            if latest_ver.major > curr_ver.major:
                return "major"
            elif latest_ver.minor > curr_ver.minor:
                return "minor"
            elif latest_ver.patch > curr_ver.patch:
                return "patch"
            else:
                return "unknown"
        except:
            # Fallback to simple comparison
            return "unknown"
    
    def _check_security_vulnerabilities(self, dep: DependencyInfo) -> None:
        """Check for known security vulnerabilities"""
        try:
            # Check with OSV (Open Source Vulnerabilities) database
            response = requests.post(
                "https://api.osv.dev/v1/query",
                json={
                    "package": {
                        "name": dep.name,
                        "ecosystem": "PyPI"
                    },
                    "version": dep.current_version
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                vulnerabilities = data.get('vulns', [])
                
                for vuln in vulnerabilities:
                    dep.security_issues.append({
                        'id': vuln.get('id'),
                        'summary': vuln.get('summary'),
                        'severity': vuln.get('database_specific', {}).get('severity', 'unknown'),
                        'fixed_version': self._extract_fixed_version(vuln)
                    })
                    
        except Exception as e:
            logger.debug(f"Error checking vulnerabilities for {dep.name}: {str(e)}")
    
    def _extract_fixed_version(self, vuln: Dict[str, Any]) -> Optional[str]:
        """Extract fixed version from vulnerability data"""
        affected = vuln.get('affected', [])
        for pkg in affected:
            if pkg.get('package', {}).get('ecosystem') == 'PyPI':
                ranges = pkg.get('ranges', [])
                for range_info in ranges:
                    events = range_info.get('events', [])
                    for event in events:
                        if 'fixed' in event:
                            return event['fixed']
        return None
    
    def update_dependencies(self, 
                          strategy: UpdateStrategy = UpdateStrategy.MINOR,
                          dry_run: bool = False) -> List[UpdateResult]:
        """Update dependencies based on strategy"""
        results = []
        dependencies = self.scan_dependencies()
        
        # Create backup if enabled
        if self.config.get('backup_enabled', True) and not dry_run:
            self._create_backup()
        
        # Process updates for each dependency type
        for dep_type, deps in dependencies.items():
            logger.info(f"Processing {dep_type} dependencies...")
            
            for dep in deps:
                if dep.update_available and self._should_update(dep, strategy):
                    logger.info(f"Updating {dep.name} from {dep.current_version} to {dep.latest_version}")
                    
                    if not dry_run:
                        result = self._perform_update(dep, dep_type)
                        results.append(result)
                        
                        if result.success:
                            # Run tests if configured
                            if self.config.get('test_command'):
                                tests_passed = self._run_tests()
                                result.tests_passed = tests_passed
                                
                                if not tests_passed and self.config.get('rollback_on_failure', True):
                                    self._rollback_update(dep, dep_type)
                                    result.rollback_performed = True
                    else:
                        # Dry run - just log what would be done
                        logger.info(f"[DRY RUN] Would update {dep.name} from {dep.current_version} to {dep.latest_version}")
        
        # Generate update report
        self._generate_update_report(results)
        
        return results
    
    def _should_update(self, dep: DependencyInfo, strategy: UpdateStrategy) -> bool:
        """Determine if dependency should be updated based on strategy"""
        # Check if package is excluded
        if dep.name in self.config.get('excluded_packages', []):
            return False
        
        # Check if package is pinned
        pinned = self.config.get('pinned_packages', {})
        if dep.name in pinned:
            return False
        
        # Check update strategy
        if strategy == UpdateStrategy.SECURITY:
            return bool(dep.security_issues)
        elif strategy == UpdateStrategy.PATCH:
            return dep.update_type == "patch"
        elif strategy == UpdateStrategy.MINOR:
            return dep.update_type in ["patch", "minor"]
        elif strategy == UpdateStrategy.MAJOR:
            return True
        elif strategy == UpdateStrategy.STABLE:
            # Check if latest version is stable (no pre-release)
            if dep.latest_version:
                try:
                    ver = version.parse(dep.latest_version)
                    return not ver.is_prerelease
                except:
                    return True
        
        return True
    
    def _create_backup(self) -> None:
        """Create backup of dependency files"""
        backup_dir = self.project_root / ".dependency_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup files
        files_to_backup = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-minimal.txt",
            "pyproject.toml",
            "environment.yml",
            "package.json",
            "package-lock.json",
            "Dockerfile"
        ]
        
        for file_name in files_to_backup:
            file_path = self.project_root / file_name
            if file_path.exists():
                shutil.copy2(file_path, backup_dir / file_name)
        
        logger.info(f"Backup created at {backup_dir}")
    
    def _perform_update(self, dep: DependencyInfo, dep_type: str) -> UpdateResult:
        """Perform the actual dependency update"""
        try:
            if dep_type == "python":
                self._update_python_dependency(dep)
            elif dep_type == "npm":
                self._update_npm_dependency(dep)
            elif dep_type == "docker":
                self._update_docker_dependency(dep)
            
            return UpdateResult(
                dependency=dep.name,
                old_version=dep.current_version,
                new_version=dep.latest_version,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error updating {dep.name}: {str(e)}")
            return UpdateResult(
                dependency=dep.name,
                old_version=dep.current_version,
                new_version=dep.latest_version,
                success=False,
                error_message=str(e)
            )
    
    def _update_python_dependency(self, dep: DependencyInfo) -> None:
        """Update Python dependency in requirements files"""
        # Update requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            self._update_requirements_file(req_file, dep)
        
        # Update other requirements files
        for req_file in ["requirements-dev.txt", "requirements-minimal.txt"]:
            file_path = self.project_root / req_file
            if file_path.exists():
                self._update_requirements_file(file_path, dep)
        
        # Update pyproject.toml if exists
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            self._update_pyproject_toml(pyproject_file, dep)
    
    def _update_requirements_file(self, file_path: Path, dep: DependencyInfo) -> None:
        """Update a requirements.txt file"""
        lines = []
        updated = False
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    try:
                        req = Requirement(line.strip())
                        if req.name.lower() == dep.name.lower():
                            # Update version
                            new_line = f"{dep.name}=={dep.latest_version}\n"
                            lines.append(new_line)
                            updated = True
                        else:
                            lines.append(line)
                    except:
                        lines.append(line)
                else:
                    lines.append(line)
        
        if updated:
            with open(file_path, 'w') as f:
                f.writelines(lines)
    
    def _update_pyproject_toml(self, file_path: Path, dep: DependencyInfo) -> None:
        """Update pyproject.toml file"""
        with open(file_path, 'r') as f:
            data = toml.load(f)
        
        # Update dependencies
        dependencies = data.get('project', {}).get('dependencies', [])
        for i, dep_str in enumerate(dependencies):
            try:
                req = Requirement(dep_str)
                if req.name.lower() == dep.name.lower():
                    dependencies[i] = f"{dep.name}=={dep.latest_version}"
            except:
                pass
        
        # Write back
        with open(file_path, 'w') as f:
            toml.dump(data, f)
    
    def _update_npm_dependency(self, dep: DependencyInfo) -> None:
        """Update NPM dependency"""
        # Use npm update command
        subprocess.run(
            ['npm', 'install', f"{dep.name}@{dep.latest_version}", '--save-exact'],
            cwd=self.project_root,
            check=True
        )
    
    def _update_docker_dependency(self, dep: DependencyInfo) -> None:
        """Update Docker base image"""
        dockerfiles = list(self.project_root.glob("**/Dockerfile*"))
        
        for dockerfile in dockerfiles:
            lines = []
            updated = False
            
            with open(dockerfile, 'r') as f:
                for line in f:
                    if line.strip().startswith('FROM'):
                        # Parse FROM statement
                        match = re.match(r'^(FROM\s+)([^\s]+)(.*)$', line)
                        if match:
                            prefix = match.group(1)
                            image = match.group(2)
                            suffix = match.group(3)
                            
                            if ':' in image:
                                image_name, _ = image.rsplit(':', 1)
                            else:
                                image_name = image
                            
                            if image_name == dep.name:
                                new_line = f"{prefix}{dep.name}:{dep.latest_version}{suffix}"
                                lines.append(new_line)
                                updated = True
                            else:
                                lines.append(line)
                        else:
                            lines.append(line)
                    else:
                        lines.append(line)
            
            if updated:
                with open(dockerfile, 'w') as f:
                    f.writelines(lines)
    
    def _run_tests(self) -> bool:
        """Run project tests"""
        test_command = self.config.get('test_command', 'python -m pytest')
        
        try:
            logger.info(f"Running tests: {test_command}")
            result = subprocess.run(
                test_command.split(),
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Tests passed successfully")
                return True
            else:
                logger.error(f"Tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return False
    
    def _rollback_update(self, dep: DependencyInfo, dep_type: str) -> None:
        """Rollback a failed update"""
        logger.warning(f"Rolling back update for {dep.name}")
        
        # Find latest backup
        backup_dir = sorted(
            (self.project_root / ".dependency_backups").glob("*"),
            reverse=True
        )[0]
        
        # Restore files based on dependency type
        if dep_type == "python":
            for file_name in ["requirements.txt", "requirements-dev.txt", "requirements-minimal.txt", "pyproject.toml"]:
                backup_file = backup_dir / file_name
                if backup_file.exists():
                    shutil.copy2(backup_file, self.project_root / file_name)
        elif dep_type == "npm":
            for file_name in ["package.json", "package-lock.json"]:
                backup_file = backup_dir / file_name
                if backup_file.exists():
                    shutil.copy2(backup_file, self.project_root / file_name)
        elif dep_type == "docker":
            backup_file = backup_dir / "Dockerfile"
            if backup_file.exists():
                shutil.copy2(backup_file, self.project_root / "Dockerfile")
    
    def _generate_update_report(self, results: List[UpdateResult]) -> None:
        """Generate update report"""
        report_dir = self.project_root / "reports" / "dependency_updates"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"update_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_updates': len(results),
            'successful_updates': len([r for r in results if r.success]),
            'failed_updates': len([r for r in results if not r.success]),
            'rollbacks': len([r for r in results if r.rollback_performed]),
            'updates': [
                {
                    'dependency': r.dependency,
                    'old_version': r.old_version,
                    'new_version': r.new_version,
                    'success': r.success,
                    'error': r.error_message,
                    'tests_passed': r.tests_passed,
                    'rollback': r.rollback_performed,
                    'time': r.update_time.isoformat()
                }
                for r in results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Update report generated: {report_file}")
    
    def check_security_vulnerabilities(self) -> Dict[str, List[DependencyInfo]]:
        """Check all dependencies for security vulnerabilities"""
        dependencies = self.scan_dependencies()
        vulnerable_deps = {}
        
        for dep_type, deps in dependencies.items():
            vulnerable = [dep for dep in deps if dep.security_issues]
            if vulnerable:
                vulnerable_deps[dep_type] = vulnerable
        
        return vulnerable_deps
    
    def generate_dependency_graph(self, output_file: str = "dependency_graph.png") -> None:
        """Generate visual dependency graph"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            G = nx.DiGraph()
            dependencies = self.scan_dependencies()
            
            # Add nodes and edges
            for dep_type, deps in dependencies.items():
                for dep in deps:
                    node_id = f"{dep_type}:{dep.name}"
                    color = 'red' if dep.security_issues else ('yellow' if dep.update_available else 'green')
                    G.add_node(node_id, color=color, type=dep_type)
            
            # Layout and draw
            pos = nx.spring_layout(G)
            colors = [G.nodes[n]['color'] for n in G.nodes()]
            
            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, node_color=colors, with_labels=True, 
                   node_size=3000, font_size=8, font_weight='bold')
            
            plt.title("Dependency Graph")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dependency graph saved to {output_file}")
            
        except ImportError:
            logger.warning("matplotlib/networkx not available for graph generation")


def main():
    """Example usage of dependency updater"""
    
    # Initialize updater
    updater = DependencyUpdater()
    
    # Scan dependencies
    logger.info("Scanning project dependencies...")
    dependencies = updater.scan_dependencies()
    
    # Print summary
    for dep_type, deps in dependencies.items():
        print(f"\n{dep_type.upper()} Dependencies:")
        for dep in deps:
            status = "🔴" if dep.security_issues else ("🟡" if dep.update_available else "🟢")
            print(f"  {status} {dep.name}: {dep.current_version} -> {dep.latest_version or 'up to date'}")
            if dep.security_issues:
                print(f"     ⚠️  Security issues: {len(dep.security_issues)}")
    
    # Check for vulnerabilities
    vulnerable = updater.check_security_vulnerabilities()
    if vulnerable:
        print("\n⚠️  SECURITY VULNERABILITIES FOUND:")
        for dep_type, deps in vulnerable.items():
            for dep in deps:
                print(f"  - {dep.name} ({dep.current_version}): {len(dep.security_issues)} issues")
    
    # Perform dry run update
    print("\nPerforming dry run update...")
    updater.update_dependencies(strategy=UpdateStrategy.MINOR, dry_run=True)


if __name__ == "__main__":
    main()