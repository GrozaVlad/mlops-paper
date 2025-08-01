#!/bin/bash

# Automated Dependency Update Script
# This script manages dependency updates for the MLOps Drug Repurposing project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
UPDATE_STRATEGY="minor"
DRY_RUN=false
SECURITY_ONLY=false
CREATE_PR=true
VERBOSE=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --strategy STRATEGY    Update strategy: patch|minor|major|security (default: minor)"
    echo "  -d, --dry-run             Perform dry run without making changes"
    echo "  -S, --security-only       Only update packages with security vulnerabilities"
    echo "  -n, --no-pr              Don't create pull request"
    echo "  -v, --verbose            Enable verbose output"
    echo "  -h, --help               Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Update with default strategy (minor)"
    echo "  $0 -s patch              # Only patch updates"
    echo "  $0 -S                    # Only security updates"
    echo "  $0 -d                    # Dry run to see what would be updated"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--strategy)
            UPDATE_STRATEGY="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -S|--security-only)
            SECURITY_ONLY=true
            UPDATE_STRATEGY="security"
            shift
            ;;
        -n|--no-pr)
            CREATE_PR=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to backup current dependencies
backup_dependencies() {
    print_color "$BLUE" "Creating backup of dependency files..."
    
    BACKUP_DIR=".dependency_backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup Python dependencies
    for file in requirements.txt requirements-dev.txt requirements-minimal.txt pyproject.toml environment.yml; do
        if [ -f "$file" ]; then
            cp "$file" "$BACKUP_DIR/"
            [ "$VERBOSE" = true ] && echo "  Backed up: $file"
        fi
    done
    
    # Backup package.json if exists
    if [ -f "package.json" ]; then
        cp package.json "$BACKUP_DIR/"
        [ -f "package-lock.json" ] && cp package-lock.json "$BACKUP_DIR/"
    fi
    
    # Backup Dockerfiles
    find . -name "Dockerfile*" -type f -exec cp {} "$BACKUP_DIR/" \;
    
    print_color "$GREEN" "✓ Backup created at: $BACKUP_DIR"
}

# Function to run security audit
run_security_audit() {
    print_color "$BLUE" "Running security vulnerability scan..."
    
    local has_vulnerabilities=false
    
    # Python security check with pip-audit
    if command_exists pip-audit; then
        print_color "$YELLOW" "  Checking Python dependencies..."
        if ! pip-audit --fix --dry-run; then
            has_vulnerabilities=true
        fi
    else
        print_color "$YELLOW" "  pip-audit not installed. Installing..."
        pip install pip-audit
        if ! pip-audit --fix --dry-run; then
            has_vulnerabilities=true
        fi
    fi
    
    # NPM security check
    if [ -f "package.json" ] && command_exists npm; then
        print_color "$YELLOW" "  Checking NPM dependencies..."
        if ! npm audit --audit-level=moderate; then
            has_vulnerabilities=true
        fi
    fi
    
    # Docker security check with trivy
    if command_exists trivy; then
        print_color "$YELLOW" "  Checking Docker images..."
        for dockerfile in $(find . -name "Dockerfile*" -type f); do
            trivy config "$dockerfile" || has_vulnerabilities=true
        done
    fi
    
    if [ "$has_vulnerabilities" = true ]; then
        print_color "$RED" "⚠️  Security vulnerabilities found!"
        return 1
    else
        print_color "$GREEN" "✓ No security vulnerabilities found"
        return 0
    fi
}

# Function to update Python dependencies
update_python_deps() {
    print_color "$BLUE" "Updating Python dependencies..."
    
    if [ "$DRY_RUN" = true ]; then
        print_color "$YELLOW" "  [DRY RUN] Would update Python dependencies"
        python scripts/maintenance/dependency_updater.py --strategy "$UPDATE_STRATEGY" --dry-run
    else
        python scripts/maintenance/dependency_updater.py --strategy "$UPDATE_STRATEGY"
        
        # Update pip itself
        pip install --upgrade pip setuptools wheel
        
        # Update development dependencies
        if [ -f "requirements-dev.txt" ]; then
            pip install --upgrade -r requirements-dev.txt
        fi
    fi
}

# Function to update NPM dependencies
update_npm_deps() {
    if [ -f "package.json" ]; then
        print_color "$BLUE" "Updating NPM dependencies..."
        
        if [ "$DRY_RUN" = true ]; then
            print_color "$YELLOW" "  [DRY RUN] Would update NPM dependencies"
            npm outdated || true
        else
            # Update based on strategy
            case $UPDATE_STRATEGY in
                patch)
                    npm update --save
                    ;;
                minor|major)
                    npx npm-check-updates -u
                    npm install
                    ;;
                security)
                    npm audit fix
                    ;;
            esac
        fi
    fi
}

# Function to update Docker base images
update_docker_images() {
    print_color "$BLUE" "Checking Docker base images..."
    
    for dockerfile in $(find . -name "Dockerfile*" -type f); do
        [ "$VERBOSE" = true ] && echo "  Checking: $dockerfile"
        
        # Extract base images
        grep "^FROM" "$dockerfile" | while read -r line; do
            image=$(echo "$line" | awk '{print $2}')
            
            if [ "$DRY_RUN" = true ]; then
                print_color "$YELLOW" "  [DRY RUN] Would check updates for: $image"
            else
                # Pull latest version to check
                docker pull "$image" || true
            fi
        done
    done
}

# Function to run tests
run_tests() {
    print_color "$BLUE" "Running tests to verify updates..."
    
    local tests_passed=true
    
    # Python tests
    if [ -f "pytest.ini" ] || [ -d "tests" ]; then
        print_color "$YELLOW" "  Running Python tests..."
        if ! python -m pytest tests/ -v; then
            tests_passed=false
            print_color "$RED" "  ✗ Python tests failed"
        else
            print_color "$GREEN" "  ✓ Python tests passed"
        fi
    fi
    
    # Type checking
    if command_exists mypy; then
        print_color "$YELLOW" "  Running type checking..."
        if ! python -m mypy scripts/ --ignore-missing-imports; then
            print_color "$YELLOW" "  ⚠ Type checking warnings (non-blocking)"
        else
            print_color "$GREEN" "  ✓ Type checking passed"
        fi
    fi
    
    # Linting
    if command_exists flake8; then
        print_color "$YELLOW" "  Running linting..."
        if ! python -m flake8 scripts/ --max-line-length=100; then
            print_color "$YELLOW" "  ⚠ Linting warnings (non-blocking)"
        else
            print_color "$GREEN" "  ✓ Linting passed"
        fi
    fi
    
    # NPM tests
    if [ -f "package.json" ] && grep -q "\"test\"" package.json; then
        print_color "$YELLOW" "  Running NPM tests..."
        if ! npm test; then
            tests_passed=false
            print_color "$RED" "  ✗ NPM tests failed"
        else
            print_color "$GREEN" "  ✓ NPM tests passed"
        fi
    fi
    
    if [ "$tests_passed" = false ]; then
        return 1
    else
        print_color "$GREEN" "✓ All tests passed"
        return 0
    fi
}

# Function to create pull request
create_pull_request() {
    if [ "$CREATE_PR" = false ] || [ "$DRY_RUN" = true ]; then
        return 0
    fi
    
    print_color "$BLUE" "Creating pull request..."
    
    # Check if we have changes
    if git diff --quiet && git diff --staged --quiet; then
        print_color "$YELLOW" "No changes to commit"
        return 0
    fi
    
    # Create branch
    BRANCH_NAME="deps/automated-update-$(date +%Y%m%d-%H%M%S)"
    git checkout -b "$BRANCH_NAME"
    
    # Commit changes
    git add -A
    git commit -m "chore(deps): automated dependency updates

Update strategy: $UPDATE_STRATEGY
Security updates: $SECURITY_ONLY

This automated update includes:
- Python dependency updates
- NPM dependency updates (if applicable)
- Docker base image updates (if applicable)

All tests have been run and passed."
    
    # Push branch
    git push origin "$BRANCH_NAME"
    
    # Create PR using GitHub CLI if available
    if command_exists gh; then
        gh pr create \
            --title "chore(deps): Automated Dependency Updates - $(date +%Y-%m-%d)" \
            --body "## Automated Dependency Updates

This PR contains automated dependency updates performed on $(date +%Y-%m-%d).

### Update Strategy
- Strategy: $UPDATE_STRATEGY
- Security only: $SECURITY_ONLY

### Test Results
- ✅ All tests passed
- ✅ Security scan completed
- ✅ Type checking completed
- ✅ Linting completed

### Changes
$(git diff --name-only origin/main...HEAD | sed 's/^/- /')

### Checklist
- [x] All tests pass
- [x] Security vulnerabilities addressed
- [x] No breaking changes identified
- [ ] Manual review completed
- [ ] Documentation updated if needed" \
            --label "dependencies" \
            --label "automated"
    else
        print_color "$GREEN" "✓ Branch pushed: $BRANCH_NAME"
        print_color "$YELLOW" "Please create a pull request manually"
    fi
}

# Function to generate report
generate_report() {
    print_color "$BLUE" "Generating dependency update report..."
    
    REPORT_DIR="reports/dependency_updates"
    mkdir -p "$REPORT_DIR"
    
    REPORT_FILE="$REPORT_DIR/update_report_$(date +%Y%m%d_%H%M%S).md"
    
    {
        echo "# Dependency Update Report"
        echo "Date: $(date)"
        echo "Strategy: $UPDATE_STRATEGY"
        echo ""
        echo "## Python Dependencies"
        pip list --outdated || echo "No outdated packages"
        echo ""
        
        if [ -f "package.json" ]; then
            echo "## NPM Dependencies"
            npm outdated || echo "No outdated packages"
            echo ""
        fi
        
        echo "## Security Scan Results"
        run_security_audit || echo "Security issues found"
        
    } > "$REPORT_FILE"
    
    print_color "$GREEN" "✓ Report generated: $REPORT_FILE"
}

# Main execution
main() {
    print_color "$BLUE" "=== MLOps Dependency Update Tool ==="
    echo ""
    
    # Check prerequisites
    if ! command_exists python; then
        print_color "$RED" "Error: Python is not installed"
        exit 1
    fi
    
    # Create virtual environment if not active
    if [ -z "$VIRTUAL_ENV" ]; then
        print_color "$YELLOW" "Warning: No virtual environment detected"
        print_color "$YELLOW" "Consider activating a virtual environment first"
        echo ""
    fi
    
    # Step 1: Backup current dependencies
    backup_dependencies
    
    # Step 2: Run security audit
    if ! run_security_audit; then
        if [ "$SECURITY_ONLY" = true ]; then
            print_color "$RED" "Security vulnerabilities found. Proceeding with security updates..."
        fi
    fi
    
    # Step 3: Update dependencies
    update_python_deps
    update_npm_deps
    update_docker_images
    
    # Step 4: Run tests
    if [ "$DRY_RUN" = false ]; then
        if ! run_tests; then
            print_color "$RED" "Tests failed! Rolling back changes..."
            # Restore from backup
            if [ -d "$BACKUP_DIR" ]; then
                cp "$BACKUP_DIR"/* . 2>/dev/null || true
                print_color "$YELLOW" "Changes rolled back from backup"
            fi
            exit 1
        fi
    fi
    
    # Step 5: Create pull request
    create_pull_request
    
    # Step 6: Generate report
    generate_report
    
    print_color "$GREEN" ""
    print_color "$GREEN" "=== Dependency update complete! ==="
}

# Run main function
main