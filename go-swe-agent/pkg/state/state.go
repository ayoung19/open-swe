package state

import (
	"time"
)

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type ToolCall struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Input    map[string]interface{} `json:"input"`
}

type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Output     string `json:"output"`
	IsError    bool   `json:"is_error,omitempty"`
}

type Plan struct {
	Tasks       []Task    `json:"tasks"`
	Summary     string    `json:"summary"`
	CreatedAt   time.Time `json:"created_at"`
	IsApproved  bool      `json:"is_approved"`
}

type Task struct {
	ID          string    `json:"id"`
	Description string    `json:"description"`
	Status      string    `json:"status"` // pending, in_progress, completed, failed
	Output      string    `json:"output,omitempty"`
	Error       string    `json:"error,omitempty"`
	StartedAt   *time.Time `json:"started_at,omitempty"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

type AgentState struct {
	Messages        []Message  `json:"messages"`
	Plan            *Plan      `json:"plan,omitempty"`
	CurrentTask     *Task      `json:"current_task,omitempty"`
	WorkingDir      string     `json:"working_dir"`
	OriginalRequest string     `json:"original_request"`
	Errors          []string   `json:"errors"`
	CompletedTasks  []Task     `json:"completed_tasks"`
}

func NewAgentState(workingDir, request string) *AgentState {
	return &AgentState{
		Messages:        []Message{},
		WorkingDir:      workingDir,
		OriginalRequest: request,
		Errors:          []string{},
		CompletedTasks:  []Task{},
	}
}

func (s *AgentState) AddMessage(role string, content interface{}) {
	s.Messages = append(s.Messages, Message{
		Role:    role,
		Content: content,
	})
}

func (s *AgentState) GetNextPendingTask() *Task {
	if s.Plan == nil {
		return nil
	}
	for i := range s.Plan.Tasks {
		if s.Plan.Tasks[i].Status == "pending" {
			return &s.Plan.Tasks[i]
		}
	}
	return nil
}

func (s *AgentState) MarkTaskComplete(taskID string, output string) {
	if s.Plan == nil {
		return
	}
	now := time.Now()
	for i := range s.Plan.Tasks {
		if s.Plan.Tasks[i].ID == taskID {
			s.Plan.Tasks[i].Status = "completed"
			s.Plan.Tasks[i].Output = output
			s.Plan.Tasks[i].CompletedAt = &now
			s.CompletedTasks = append(s.CompletedTasks, s.Plan.Tasks[i])
			break
		}
	}
}

func (s *AgentState) MarkTaskFailed(taskID string, err string) {
	if s.Plan == nil {
		return
	}
	now := time.Now()
	for i := range s.Plan.Tasks {
		if s.Plan.Tasks[i].ID == taskID {
			s.Plan.Tasks[i].Status = "failed"
			s.Plan.Tasks[i].Error = err
			s.Plan.Tasks[i].CompletedAt = &now
			s.Errors = append(s.Errors, err)
			break
		}
	}
}

func (s *AgentState) StartTask(taskID string) {
	if s.Plan == nil {
		return
	}
	now := time.Now()
	for i := range s.Plan.Tasks {
		if s.Plan.Tasks[i].ID == taskID {
			s.Plan.Tasks[i].Status = "in_progress"
			s.Plan.Tasks[i].StartedAt = &now
			s.CurrentTask = &s.Plan.Tasks[i]
			break
		}
	}
}

func (s *AgentState) AllTasksComplete() bool {
	if s.Plan == nil {
		return false
	}
	for _, task := range s.Plan.Tasks {
		if task.Status != "completed" && task.Status != "failed" {
			return false
		}
	}
	return true
}