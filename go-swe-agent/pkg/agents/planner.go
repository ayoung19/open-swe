package agents

import (
	"fmt"
	"strings"
	"time"

	"github.com/openswe/go-swe-agent/pkg/llm"
	"github.com/openswe/go-swe-agent/pkg/state"
	"github.com/openswe/go-swe-agent/pkg/tools"
)

type Planner struct {
	client       *llm.BedrockClient
	toolExecutor *tools.ToolExecutor
}

func NewPlanner(workingDir string) *Planner {
	return &Planner{
		client:       llm.NewBedrockClient(),
		toolExecutor: tools.NewToolExecutor(workingDir),
	}
}

func (p *Planner) GeneratePlan(agentState *state.AgentState) error {
	fmt.Println("\n🔍 Analyzing codebase and generating plan...")
	
	// First, gather context about the codebase
	messages := p.buildContextMessages(agentState)
	
	// Get codebase structure
	systemPrompt := p.buildPlannerSystemPrompt()
	
	// Call LLM with tools to explore the codebase
	availableTools := p.getPlannerTools()
	
	// Initial exploration
	for i := 0; i < 5; i++ { // Allow up to 5 tool calls for exploration
		response, err := p.client.CreateMessage(messages, systemPrompt, availableTools)
		if err != nil {
			return fmt.Errorf("failed to get LLM response: %w", err)
		}
		
		text, toolCalls, _ := p.client.ParseContent(response.Content)
		
		if len(toolCalls) > 0 {
			// Execute tool calls
			messages = append(messages, llm.AnthropicMessage{
				Role:    "assistant",
				Content: response.Content,
			})
			
			var toolResults []interface{}
			for _, toolCall := range toolCalls {
				fmt.Printf("  📂 Exploring: %s\n", toolCall.Name)
				output, err := p.toolExecutor.Execute(toolCall.Name, toolCall.Input)
				if err != nil {
					output = fmt.Sprintf("Error: %v", err)
				}
				
				// Truncate very long outputs
				if len(output) > 5000 {
					output = output[:5000] + "\n... (truncated)"
				}
				
				toolResults = append(toolResults, llm.ToolResultContent{
					Type:      "tool_result",
					ToolUseID: toolCall.ID,
					Content:   output,
				})
			}
			
			messages = append(messages, llm.AnthropicMessage{
				Role:    "user",
				Content: toolResults,
			})
		} else {
			// No more tool calls, we should have a plan now
			if strings.Contains(text, "PLAN:") {
				plan := p.parsePlanFromText(text)
				if plan != nil {
					agentState.Plan = plan
					fmt.Printf("\n✅ Generated plan with %d tasks\n", len(plan.Tasks))
					return nil
				}
			}
		}
	}
	
	// Final attempt to get a plan without tools
	messages = append(messages, llm.AnthropicMessage{
		Role: "user",
		Content: []interface{}{
			llm.TextContent{
				Type: "text",
				Text: "Based on your exploration, please provide a concrete plan in the format:\nPLAN:\n1. [Task description]\n2. [Task description]\n...",
			},
		},
	})
	
	response, err := p.client.CreateMessage(messages, systemPrompt, nil)
	if err != nil {
		return fmt.Errorf("failed to get final plan: %w", err)
	}
	
	text, _, _ := p.client.ParseContent(response.Content)
	plan := p.parsePlanFromText(text)
	if plan == nil {
		return fmt.Errorf("failed to generate a valid plan")
	}
	
	agentState.Plan = plan
	fmt.Printf("\n✅ Generated plan with %d tasks\n", len(plan.Tasks))
	return nil
}

func (p *Planner) buildContextMessages(agentState *state.AgentState) []llm.AnthropicMessage {
	return []llm.AnthropicMessage{
		{
			Role: "user",
			Content: []interface{}{
				llm.TextContent{
					Type: "text",
					Text: fmt.Sprintf(`Please analyze this codebase and create a detailed plan to complete the following request:

REQUEST: %s

First, explore the codebase structure to understand:
1. The project layout and key files
2. The technology stack and dependencies
3. Existing patterns and conventions
4. Relevant code sections for this task

Then provide a concrete, step-by-step plan to complete the request.`, agentState.OriginalRequest),
				},
			},
		},
	}
}

func (p *Planner) buildPlannerSystemPrompt() string {
	return `You are an expert software engineer tasked with planning code changes.

Your job is to:
1. Thoroughly analyze the codebase structure
2. Understand the existing patterns and conventions
3. Create a detailed, actionable plan to complete the requested changes

Use the available tools to explore the codebase:
- Use list_files to understand the project structure
- Use read_file to examine key files (README, package.json, go.mod, etc.)
- Use search to find relevant code patterns
- Use bash for commands like 'find', 'ls -la', etc.

After exploration, provide your plan in this format:
PLAN:
1. [Specific task description]
2. [Specific task description]
...

Each task should be concrete and actionable. Focus on:
- Understanding before changing
- Following existing patterns
- Making incremental, testable changes
- Ensuring the code remains functional`
}

func (p *Planner) getPlannerTools() []llm.Tool {
	toolDefs := tools.GetAvailableTools()
	var llmTools []llm.Tool
	
	for _, toolDef := range toolDefs {
		llmTools = append(llmTools, llm.Tool{
			Name:        toolDef["name"].(string),
			Description: toolDef["description"].(string),
			InputSchema: toolDef["input_schema"].(map[string]interface{}),
		})
	}
	
	return llmTools
}

func (p *Planner) parsePlanFromText(text string) *state.Plan {
	if !strings.Contains(text, "PLAN:") {
		return nil
	}
	
	parts := strings.Split(text, "PLAN:")
	if len(parts) < 2 {
		return nil
	}
	
	planText := parts[1]
	lines := strings.Split(planText, "\n")
	
	var tasks []state.Task
	taskID := 1
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		// Look for numbered items
		for _, prefix := range []string{"1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.", "13.", "14.", "15."} {
			if strings.HasPrefix(line, prefix) {
				taskDesc := strings.TrimSpace(strings.TrimPrefix(line, prefix))
				if taskDesc != "" {
					tasks = append(tasks, state.Task{
						ID:          fmt.Sprintf("task-%d", taskID),
						Description: taskDesc,
						Status:      "pending",
					})
					taskID++
					break
				}
			}
		}
		
		// Also handle bullet points
		if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
			taskDesc := strings.TrimSpace(line[2:])
			if taskDesc != "" {
				tasks = append(tasks, state.Task{
					ID:          fmt.Sprintf("task-%d", taskID),
					Description: taskDesc,
					Status:      "pending",
				})
				taskID++
			}
		}
	}
	
	if len(tasks) == 0 {
		return nil
	}
	
	return &state.Plan{
		Tasks:      tasks,
		Summary:    fmt.Sprintf("Plan with %d tasks", len(tasks)),
		CreatedAt:  time.Now(),
		IsApproved: true, // Auto-approve for simplicity
	}
}