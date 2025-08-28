package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

type AnthropicClient struct {
	apiKey  string
	baseURL string
	model   string
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type TextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ToolUseContent struct {
	Type  string                 `json:"type"`
	ID    string                 `json:"id"`
	Name  string                 `json:"name"`
	Input map[string]interface{} `json:"input"`
}

type ToolResultContent struct {
	Type       string `json:"type"`
	ToolUseID  string `json:"tool_use_id"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

type AnthropicRequest struct {
	Model     string              `json:"model"`
	MaxTokens int                 `json:"max_tokens"`
	Messages  []AnthropicMessage  `json:"messages"`
	System    string              `json:"system,omitempty"`
	Tools     []Tool              `json:"tools,omitempty"`
}

type AnthropicResponse struct {
	ID      string               `json:"id"`
	Type    string               `json:"type"`
	Role    string               `json:"role"`
	Content []json.RawMessage    `json:"content"`
	Model   string               `json:"model"`
	Usage   Usage                `json:"usage"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

func NewAnthropicClient() *AnthropicClient {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		panic("ANTHROPIC_API_KEY environment variable is required")
	}
	
	return &AnthropicClient{
		apiKey:  apiKey,
		baseURL: "https://api.anthropic.com/v1/messages",
		model:   "claude-3-5-sonnet-20241022",
	}
}

func (c *AnthropicClient) CreateMessage(messages []AnthropicMessage, system string, tools []Tool) (*AnthropicResponse, error) {
	req := AnthropicRequest{
		Model:     c.model,
		MaxTokens: 8192,
		Messages:  messages,
		System:    system,
		Tools:     tools,
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", c.baseURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var anthropicResp AnthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &anthropicResp, nil
}

func (c *AnthropicClient) ParseContent(content []json.RawMessage) (string, []ToolUseContent, error) {
	var text string
	var toolCalls []ToolUseContent

	for _, raw := range content {
		var base map[string]interface{}
		if err := json.Unmarshal(raw, &base); err != nil {
			continue
		}

		contentType, ok := base["type"].(string)
		if !ok {
			continue
		}

		switch contentType {
		case "text":
			if textVal, ok := base["text"].(string); ok {
				text += textVal
			}
		case "tool_use":
			var toolUse ToolUseContent
			if err := json.Unmarshal(raw, &toolUse); err == nil {
				toolCalls = append(toolCalls, toolUse)
			}
		}
	}

	return text, toolCalls, nil
}