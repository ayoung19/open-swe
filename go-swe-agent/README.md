# Go SWE Agent

A simplified autonomous coding agent written in Go that can plan and execute complex coding tasks on any codebase. Uses AWS Bedrock with Claude 3 Opus for superior reasoning capabilities.

## Features

- 🔍 **Automatic Codebase Analysis**: Explores and understands your project structure
- 📋 **Intelligent Planning**: Creates step-by-step plans for complex tasks using Claude 3 Opus
- 🔧 **Autonomous Execution**: Executes tasks using file operations and shell commands
- 🎯 **Focused Simplicity**: Minimal dependencies, maximum effectiveness
- ☁️ **AWS Bedrock Integration**: Leverages enterprise-grade AI infrastructure

## Prerequisites

1. **AWS Account** with Amazon Bedrock access
2. **Claude 3 Opus** model enabled in your AWS Bedrock console
3. **AWS CLI** configured with appropriate credentials

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

## Setup

### Configure AWS Credentials:

#### Option 1: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-west-2  # Optional, defaults to us-west-2
```

#### Option 2: AWS CLI Configuration
```bash
aws configure
```

### Enable Claude 3 Opus in AWS Bedrock:
1. Go to AWS Bedrock console
2. Navigate to Model access
3. Request access to "Claude 3 Opus" (anthropic.claude-3-opus-20240229)
4. Wait for approval (usually instant for Claude models)

## Usage

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
- AWS Account with Bedrock access
- Claude 3 Opus model enabled in AWS Bedrock
- Unix-like environment (Linux, macOS, WSL)

## Limitations

- Currently only supports AWS Bedrock with Claude 3 Opus
- Requires environment with bash shell
- No built-in rollback mechanism (use version control)
- AWS region must have Bedrock available

## Contributing

This is a simplified implementation focusing on core functionality. Contributions are welcome to:
- Add support for more LLM providers
- Improve error handling and recovery
- Add more sophisticated planning strategies
- Enhance tool capabilities

## License

MIT