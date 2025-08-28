package agents

import (
	"fmt"
	"strings"

	"github.com/fatih/color"
	"github.com/openswe/go-swe-agent/pkg/llm"
	"github.com/openswe/go-swe-agent/pkg/state"
	"github.com/openswe/go-swe-agent/pkg/tools"
)

type Executor struct {
	client       *llm.AnthropicClient
	toolExecutor *tools.ToolExecutor
}

func NewExecutor(workingDir string) *Executor {
	return &Executor{
		client:       llm.NewAnthropicClient(),
		toolExecutor: tools.NewToolExecutor(workingDir),
	}
}

func (e *Executor) ExecuteTask(agentState *state.AgentState, task *state.Task) error {
	color.Yellow("\n🔧 Executing: %s\n", task.Description)
	
	agentState.StartTask(task.ID)
	
	// Build conversation with task context
	messages := e.buildTaskMessages(agentState, task)
	systemPrompt := e.buildExecutorSystemPrompt()
	availableTools := e.getExecutorTools()
	
	// Allow up to 15 iterations for complex tasks
	maxIterations := 15
	for i := 0; i < maxIterations; i++ {
		response, err := e.client.CreateMessage(messages, systemPrompt, availableTools)
		if err != nil {
			agentState.MarkTaskFailed(task.ID, err.Error())
			return fmt.Errorf("LLM error: %w", err)
		}
		
		text, toolCalls, _ := e.client.ParseContent(response.Content)
		
		// Add assistant message
		messages = append(messages, llm.AnthropicMessage{
			Role:    "assistant",
			Content: response.Content,
		})
		
		if len(toolCalls) > 0 {
			// Execute tool calls
			var toolResults []interface{}
			
			for _, toolCall := range toolCalls {
				color.Cyan("  🔨 %s: %s\n", toolCall.Name, e.getToolDescription(toolCall))
				
				output, err := e.toolExecutor.Execute(toolCall.Name, toolCall.Input)
				isError := err != nil
				
				if err != nil {
					output = fmt.Sprintf("Error: %v", err)
				}
				
				// Truncate very long outputs
				if len(output) > 10000 {
					output = output[:10000] + "\n... (output truncated)"
				}
				
				toolResults = append(toolResults, llm.ToolResultContent{
					Type:      "tool_result",
					ToolUseID: toolCall.ID,
					Content:   output,
					IsError:   isError,
				})
			}
			
			messages = append(messages, llm.AnthropicMessage{
				Role:    "user",
				Content: toolResults,
			})
			
		} else if strings.Contains(strings.ToLower(text), "task completed") || 
				  strings.Contains(strings.ToLower(text), "task complete") ||
				  strings.Contains(strings.ToLower(text), "successfully completed") ||
				  strings.Contains(strings.ToLower(text), "done") && i > 0 {
			// Task completed successfully
			agentState.MarkTaskComplete(task.ID, text)
			color.Green("  ✅ Task completed\n")
			return nil
		} else if i == 0 && text != "" {
			// First response with no tools, ask to proceed
			messages = append(messages, llm.AnthropicMessage{
				Role: "user",
				Content: []interface{}{
					llm.TextContent{
						Type: "text",
						Text: "Please proceed with implementing this task using the available tools.",
					},
				},
			})
		}
	}
	
	// Max iterations reached
	agentState.MarkTaskComplete(task.ID, "Task completed (max iterations reached)")
	return nil
}

func (e *Executor) buildTaskMessages(agentState *state.AgentState, task *state.Task) []llm.AnthropicMessage {
	// Build context from completed tasks
	var context strings.Builder
	if len(agentState.CompletedTasks) > 0 {
		context.WriteString("Previously completed tasks:\n")
		for _, t := range agentState.CompletedTasks {
			context.WriteString(fmt.Sprintf("- %s\n", t.Description))
		}
		context.WriteString("\n")
	}
	
	return []llm.AnthropicMessage{
		{
			Role: "user",
			Content: []interface{}{
				llm.TextContent{
					Type: "text",
					Text: fmt.Sprintf(`%sCurrent task to implement:
%s

Original request context: %s

Please implement this task step by step. Use the available tools to:
1. Read relevant files to understand the code
2. Make necessary changes
3. Test your changes if applicable
4. Verify the implementation

When the task is complete, say "Task completed" with a brief summary.`, 
						context.String(), task.Description, agentState.OriginalRequest),
				},
			},
		},
	}
}

func (e *Executor) buildExecutorSystemPrompt() string {
	return `You are an expert software engineer implementing specific tasks.

Your approach should be:
1. First understand the existing code by reading relevant files
2. Follow existing patterns and conventions in the codebase
3. Make changes incrementally and test when possible
4. Ensure your changes don't break existing functionality
5. Write clean, maintainable code

Important guidelines:
- Always read before writing to understand context
- Follow the existing code style and patterns
- Test your changes when possible using bash commands
- Create directories before writing files to them
- Handle errors gracefully
- When task is complete, explicitly state "Task completed" with a summary

Be thorough but efficient. Focus on correctness over speed.`
}

func (e *Executor) getExecutorTools() []llm.Tool {
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

func (e *Executor) getToolDescription(toolCall llm.ToolUseContent) string {
	switch toolCall.Name {
	case "bash":
		if cmd, ok := toolCall.Input["command"].(string); ok {
			// Truncate long commands
			if len(cmd) > 100 {
				return cmd[:100] + "..."
			}
			return cmd
		}
	case "read_file":
		if path, ok := toolCall.Input["path"].(string); ok {
			return path
		}
	case "write_file":
		if path, ok := toolCall.Input["path"].(string); ok {
			return path
		}
	case "search":
		if pattern, ok := toolCall.Input["pattern"].(string); ok {
			return fmt.Sprintf("'%s'", pattern)
		}
	case "list_files":
		if path, ok := toolCall.Input["path"].(string); ok {
			return path
		}
		return "current directory"
	}
	return ""
}