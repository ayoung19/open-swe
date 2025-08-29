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
	// Check for AWS credentials
	if os.Getenv("AWS_ACCESS_KEY_ID") == "" || os.Getenv("AWS_SECRET_ACCESS_KEY") == "" {
		color.Red("Error: AWS credentials are required\n")
		fmt.Println("\nPlease configure your AWS credentials:")
		fmt.Println("  export AWS_ACCESS_KEY_ID=your-access-key")
		fmt.Println("  export AWS_SECRET_ACCESS_KEY=your-secret-key")
		fmt.Println("  export AWS_REGION=us-west-2  # Optional, defaults to us-west-2")
		fmt.Println("\nOr configure using AWS CLI:")
		fmt.Println("  aws configure")
		fmt.Println("\nMake sure your AWS account has access to Amazon Bedrock with Claude 3 Opus model.")
		os.Exit(1)
	}

	// Create and run orchestrator
	orchestrator := graph.NewOrchestrator(workingDir, request)
	
	if err := orchestrator.Run(); err != nil {
		color.Red("\n❌ Agent failed: %v\n", err)
		os.Exit(1)
	}
}