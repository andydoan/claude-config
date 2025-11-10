# Claude Code Setup

Personal Claude Code configuration for productive web development with **12 slash commands** and **11 specialized AI agents**.

## What's Inside

### 📋 Development Commands (7)

- `/new-task` - Analyze code for performance issues
- `/code-explain` - Generate detailed explanations
- `/code-optimize` - Performance optimization
- `/code-cleanup` - Refactoring and cleanup
- `/feature-plan` - Feature implementation planning
- `/lint` - Linting and fixes
- `/docs-generate` - Documentation generation

### 🔌 API Commands (3)

- `/api-new` - Create new API endpoints
- `/api-test` - Test API endpoints
- `/api-protect` - Add protection & validation

### 🎨 UI Commands (2)

- `/component-new` - Create React components
- `/page-new` - Create Next.js pages

### 🤖 Specialized AI Agents (11)

**Architecture & Planning**
- **tech-stack-researcher** - Technology choice recommendations with trade-offs
- **system-architect** - Scalable system architecture design
- **backend-architect** - Backend systems with data integrity & security
- **frontend-architect** - Performant, accessible UI architecture
- **requirements-analyst** - Transform ideas into concrete specifications

**Code Quality & Performance**
- **refactoring-expert** - Systematic refactoring and clean code
- **performance-engineer** - Measurement-driven optimization
- **security-engineer** - Vulnerability identification and security standards

**Documentation & Research**
- **technical-writer** - Clear, comprehensive documentation
- **learning-guide** - Teaching programming concepts progressively
- **deep-research-agent** - Comprehensive research with adaptive strategies

## Best For

- Next.js developers
- TypeScript projects
- React developers
- Full-stack engineers

## Usage Examples

### Planning a Feature

```bash
/feature-plan
# Then describe your feature idea
```

### Creating an API

```bash
/api-new
# Claude will scaffold a complete API route with types, validation, and error handling
```

### Research Tech Choices

Just ask Claude questions like:
- "Should I use WebSockets or SSE?"
- "How should I structure this database?"
- "What's the best library for X?"

The tech-stack-researcher agent automatically activates and provides detailed, researched answers.

## Philosophy

This setup emphasizes:
- **Type Safety**: Never uses `any` types
- **Best Practices**: Follows modern Next.js/React patterns
- **Productivity**: Reduces repetitive scaffolding
- **Research**: AI-powered tech decisions with evidence

## Customization

Customize commands and agents by editing files in `.claude/commands/` and `.claude/agents/`.
