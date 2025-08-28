# Go SWE Agent

A simplified autonomous coding agent written in Go that can plan and execute complex coding tasks on any codebase.

## Features

- 🔍 **Automatic Codebase Analysis**: Explores and understands your project structure
- 📋 **Intelligent Planning**: Creates step-by-step plans for complex tasks
- 🔧 **Autonomous Execution**: Executes tasks using file operations and shell commands
- 🎯 **Focused Simplicity**: Minimal dependencies, maximum effectiveness

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd go-swe-agent

# Install dependencies
go mod download

# Build the agent
go build -o go-swe-agent cmd/main.go
```

## Usage

### Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your-api-key-here
```

### Run the agent:
```bash
# Basic usage
./go-swe-agent --request "Add a REST API endpoint for user management"

# Specify a different directory
./go-swe-agent --dir ./my-project --request "Fix the authentication bug"

# Short flags
./go-swe-agent -d ./src -r "Refactor the database layer to use connection pooling"
```

## Examples

### Add a new feature:
```bash
./go-swe-agent -r "Add a caching layer to the API using Redis"
```

### Fix a bug:
```bash
./go-swe-agent -r "Fix the null pointer exception in the user service"
```

### Refactor code:
```bash
./go-swe-agent -r "Refactor the authentication system to use JWT tokens"
```

### Add tests:
```bash
./go-swe-agent -r "Add unit tests for the payment processing module"
```

## How It Works

1. **Planning Phase**: The agent analyzes your codebase, reads relevant files, and creates a detailed plan
2. **Execution Phase**: Each task in the plan is executed sequentially using available tools
3. **Verification**: The agent verifies changes and can run tests if needed

## Available Tools

The agent has access to:
- **bash**: Execute shell commands
- **read_file**: Read file contents
- **write_file**: Create or modify files
- **list_files**: List directory contents
- **search**: Search for patterns in files (uses ripgrep/grep)

## Architecture

```
go-swe-agent/
├── cmd/
│   └── main.go           # CLI entry point
├── pkg/
│   ├── agents/
│   │   ├── planner.go    # Planning logic
│   │   └── executor.go   # Task execution logic
│   ├── graph/
│   │   └── orchestrator.go # Main orchestration
│   ├── llm/
│   │   └── anthropic.go  # Anthropic API client
│   ├── state/
│   │   └── state.go      # State management
│   └── tools/
│       └── tools.go      # Tool implementations
```

## Requirements

- Go 1.21 or higher
- Anthropic API key
- Unix-like environment (Linux, macOS, WSL)

## Limitations

- Currently only supports Anthropic's Claude API
- Requires environment with bash shell
- No built-in rollback mechanism (use version control)

## Contributing

This is a simplified implementation focusing on core functionality. Contributions are welcome to:
- Add support for more LLM providers
- Improve error handling and recovery
- Add more sophisticated planning strategies
- Enhance tool capabilities

## License

MIT