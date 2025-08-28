package graph

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/fatih/color"
	"github.com/openswe/go-swe-agent/pkg/agents"
	"github.com/openswe/go-swe-agent/pkg/state"
)

type Orchestrator struct {
	state    *state.AgentState
	planner  *agents.Planner
	executor *agents.Executor
}

func NewOrchestrator(workingDir, request string) *Orchestrator {
	// Resolve to absolute path
	absPath, err := filepath.Abs(workingDir)
	if err != nil {
		absPath = workingDir
	}
	
	return &Orchestrator{
		state:    state.NewAgentState(absPath, request),
		planner:  agents.NewPlanner(absPath),
		executor: agents.NewExecutor(absPath),
	}
}

func (o *Orchestrator) Run() error {
	color.Blue("\n═══════════════════════════════════════════")
	color.Blue("       🤖 Go SWE Agent Starting")
	color.Blue("═══════════════════════════════════════════\n")
	
	fmt.Printf("📁 Working Directory: %s\n", o.state.WorkingDir)
	fmt.Printf("📝 Request: %s\n", o.state.OriginalRequest)
	
	// Verify working directory exists
	if _, err := os.Stat(o.state.WorkingDir); os.IsNotExist(err) {
		return fmt.Errorf("working directory does not exist: %s", o.state.WorkingDir)
	}
	
	// Phase 1: Planning
	color.Yellow("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	color.Yellow("  Phase 1: Planning")
	color.Yellow("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	
	if err := o.planner.GeneratePlan(o.state); err != nil {
		return fmt.Errorf("planning failed: %w", err)
	}
	
	if o.state.Plan == nil || len(o.state.Plan.Tasks) == 0 {
		return fmt.Errorf("no plan generated")
	}
	
	// Display the plan
	o.displayPlan()
	
	// Phase 2: Execution
	color.Yellow("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	color.Yellow("  Phase 2: Execution")
	color.Yellow("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	
	// Execute each task
	for i := range o.state.Plan.Tasks {
		fmt.Printf("\n[%d/%d] ", i+1, len(o.state.Plan.Tasks))
		
		if err := o.executor.ExecuteTask(o.state, &o.state.Plan.Tasks[i]); err != nil {
			color.Red("  ❌ Task failed: %v\n", err)
			// Continue with other tasks even if one fails
			continue
		}
	}
	
	// Final summary
	o.displaySummary()
	
	return nil
}

func (o *Orchestrator) displayPlan() {
	color.Green("\n📋 Generated Plan:\n")
	color.Green("─────────────────\n")
	
	for i, task := range o.state.Plan.Tasks {
		fmt.Printf("%d. %s\n", i+1, task.Description)
	}
	
	fmt.Printf("\nTotal tasks: %d\n", len(o.state.Plan.Tasks))
}

func (o *Orchestrator) displaySummary() {
	color.Blue("\n═══════════════════════════════════════════")
	color.Blue("       📊 Execution Summary")
	color.Blue("═══════════════════════════════════════════\n")
	
	completed := 0
	failed := 0
	pending := 0
	
	for _, task := range o.state.Plan.Tasks {
		switch task.Status {
		case "completed":
			completed++
		case "failed":
			failed++
		case "pending":
			pending++
		}
	}
	
	color.Green("  ✅ Completed: %d\n", completed)
	if failed > 0 {
		color.Red("  ❌ Failed: %d\n", failed)
	}
	if pending > 0 {
		color.Yellow("  ⏳ Pending: %d\n", pending)
	}
	
	if len(o.state.Errors) > 0 {
		color.Red("\n⚠️  Errors encountered:\n")
		for _, err := range o.state.Errors {
			fmt.Printf("  - %s\n", err)
		}
	}
	
	if completed == len(o.state.Plan.Tasks) {
		color.Green("\n🎉 All tasks completed successfully!\n")
	} else if completed > 0 {
		color.Yellow("\n⚡ Partial completion: %d/%d tasks done\n", completed, len(o.state.Plan.Tasks))
	}
}