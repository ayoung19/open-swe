package main

import (
	"fmt"
	"os"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
	"github.com/openswe/go-swe-agent/pkg/graph"
)

var (
	workingDir string
	request    string
)

func main() {
	var rootCmd = &cobra.Command{
		Use:   "go-swe-agent",
		Short: "A simple autonomous coding agent",
		Long: `Go SWE Agent - An autonomous coding agent that plans and executes tasks.
		
This agent will:
1. Analyze your codebase
2. Create a plan to complete your request
3. Execute the plan step by step

Example:
  go-swe-agent --dir ./my-project --request "Add a new REST API endpoint for user management"
  go-swe-agent -d . -r "Fix the bug in the authentication system"`,
		Run: runAgent,
	}

	rootCmd.Flags().StringVarP(&workingDir, "dir", "d", ".", "Working directory for the agent")
	rootCmd.Flags().StringVarP(&request, "request", "r", "", "The task request for the agent")
	rootCmd.MarkFlagRequired("request")

	if err := rootCmd.Execute(); err != nil {
		color.Red("Error: %v\n", err)
		os.Exit(1)
	}
}

func runAgent(cmd *cobra.Command, args []string) {
	// Check for API key
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		color.Red("Error: ANTHROPIC_API_KEY environment variable is required\n")
		fmt.Println("\nPlease set your Anthropic API key:")
		fmt.Println("  export ANTHROPIC_API_KEY=your-api-key")
		os.Exit(1)
	}

	// Create and run orchestrator
	orchestrator := graph.NewOrchestrator(workingDir, request)
	
	if err := orchestrator.Run(); err != nil {
		color.Red("\n❌ Agent failed: %v\n", err)
		os.Exit(1)
	}
}