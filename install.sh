#!/usr/bin/env bash

# Claude Code Configuration Installer
# Installs custom commands and agents to your Claude setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR="${SCRIPT_DIR}/.claude"

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Claude Code Configuration Installer${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if .claude source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source .claude directory not found at: $SOURCE_DIR"
    exit 1
fi

# Determine installation target
INSTALL_TYPE=""
TARGET_DIR=""

echo "Where would you like to install the Claude configuration?"
echo ""
echo "  1) Current directory (project-specific)"
echo "  2) Home directory (global, all projects)"
echo "  3) Custom path"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        INSTALL_TYPE="project"
        TARGET_DIR="$(pwd)/.claude"
        ;;
    2)
        INSTALL_TYPE="global"
        TARGET_DIR="$HOME/.claude"
        ;;
    3)
        read -p "Enter custom path: " custom_path
        INSTALL_TYPE="custom"
        TARGET_DIR="${custom_path}/.claude"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_info "Installation target: $TARGET_DIR"
echo ""

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Check for existing agents/commands and prompt for backup
NEEDS_BACKUP=false
if [ -d "$TARGET_DIR/agents" ] || [ -d "$TARGET_DIR/commands" ]; then
    print_warning "Existing commands or agents found in: $TARGET_DIR"

    if [ -d "$TARGET_DIR/agents" ]; then
        agent_count=$(find "$TARGET_DIR/agents" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
        echo "  - agents/ (${agent_count} agents)"
    fi

    if [ -d "$TARGET_DIR/commands" ]; then
        command_count=$(find "$TARGET_DIR/commands" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
        echo "  - commands/ (${command_count} commands)"
    fi

    echo ""
    read -p "Backup existing and continue? [y/N]: " backup_choice

    if [[ ! $backup_choice =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    NEEDS_BACKUP=true
fi

# Copy configuration files
print_info "Installing configuration files..."

# Copy agents
if [ -d "$SOURCE_DIR/agents" ]; then
    # Backup existing agents if needed
    if [ "$NEEDS_BACKUP" = true ] && [ -d "$TARGET_DIR/agents" ]; then
        BACKUP_DIR="${TARGET_DIR}/agents.backup.$(date +%Y%m%d_%H%M%S)"
        mv "$TARGET_DIR/agents" "$BACKUP_DIR"
        print_success "Backed up existing agents to: agents.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    cp -r "$SOURCE_DIR/agents" "$TARGET_DIR/"
    agent_count=$(find "$SOURCE_DIR/agents" -name "*.md" | wc -l | tr -d ' ')
    print_success "Installed ${agent_count} agents"
fi

# Copy commands
if [ -d "$SOURCE_DIR/commands" ]; then
    # Backup existing commands if needed
    if [ "$NEEDS_BACKUP" = true ] && [ -d "$TARGET_DIR/commands" ]; then
        BACKUP_DIR="${TARGET_DIR}/commands.backup.$(date +%Y%m%d_%H%M%S)"
        mv "$TARGET_DIR/commands" "$BACKUP_DIR"
        print_success "Backed up existing commands to: commands.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    cp -r "$SOURCE_DIR/commands" "$TARGET_DIR/"
    command_count=$(find "$SOURCE_DIR/commands" -name "*.md" | wc -l | tr -d ' ')
    print_success "Installed ${command_count} commands"
fi

# Copy settings template if it doesn't exist
if [ -f "$SOURCE_DIR/settings.template.json" ]; then
    if [ ! -f "$TARGET_DIR/settings.local.json" ]; then
        cp "$SOURCE_DIR/settings.template.json" "$TARGET_DIR/settings.local.json"
        print_success "Created settings.local.json (customize this file)"
    else
        print_info "Keeping existing settings.local.json"
    fi
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Installation Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""

# Print summary
print_info "Configuration installed to: $TARGET_DIR"
echo ""
echo "Available commands:"
echo "  • /new-task - Analyze task complexity"
echo "  • /code-optimize - Performance optimization"
echo "  • /code-cleanup - Refactoring and cleanup"
echo "  • /feature-plan - Feature planning"
echo "  • /lint - Linting and fixes"
echo "  • /docs-generate - Documentation generation"
echo "  • /api-new - Create API endpoints"
echo "  • /api-test - Test API endpoints"
echo "  • /api-protect - Add API protection"
echo "  • /component-new - Create React components"
echo "  • /page-new - Create Next.js pages"
echo ""
echo "Specialized agents are automatically activated based on context."
echo ""

# Next steps
if [ "$INSTALL_TYPE" = "project" ]; then
    print_info "This is a project-specific installation"
    print_warning "Add .claude/settings.local.json to your .gitignore"
elif [ "$INSTALL_TYPE" = "global" ]; then
    print_info "This is a global installation (affects all projects)"
fi

echo ""
print_success "Ready to use! Open Claude Code and start using your new commands."
echo ""
