package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// BedrockClient implements the same interface as AnthropicClient but uses AWS Bedrock
type BedrockClient struct {
	client  *bedrockruntime.Client
	model   string
	region  string
}

// BedrockRequest matches Anthropic's API format for easier compatibility
type BedrockRequest struct {
	AnthropicVersion string             `json:"anthropic_version"`
	MaxTokens        int                `json:"max_tokens"`
	Messages         []AnthropicMessage `json:"messages"`
	System           string             `json:"system,omitempty"`
	Tools            []Tool             `json:"tools,omitempty"`
}

// BedrockResponse matches Anthropic's response format
type BedrockResponse struct {
	ID      string            `json:"id"`
	Type    string            `json:"type"`
	Role    string            `json:"role"`
	Content []json.RawMessage `json:"content"`
	Model   string            `json:"model"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func NewBedrockClient() *BedrockClient {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = "us-west-2" // Default region
	}

	cfg, err := config.LoadDefaultConfig(context.TODO(),
		config.WithRegion(region),
	)
	if err != nil {
		panic(fmt.Sprintf("Failed to load AWS config: %v", err))
	}

	return &BedrockClient{
		client: bedrockruntime.NewFromConfig(cfg),
		model:  "anthropic.claude-3-opus-20240229",
		region: region,
	}
}

// CreateMessage sends a message to Bedrock using the same interface as AnthropicClient
func (c *BedrockClient) CreateMessage(messages []AnthropicMessage, system string, tools []Tool) (*AnthropicResponse, error) {
	// Build the request in Anthropic format
	req := BedrockRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        8192,
		Messages:         messages,
		System:           system,
		Tools:            tools,
	}

	// Marshal the request
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call Bedrock InvokeModel API
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(c.model),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        jsonData,
	}

	resp, err := c.client.InvokeModel(context.TODO(), input)
	if err != nil {
		return nil, fmt.Errorf("bedrock invoke error: %w", err)
	}

	// Parse the response
	var bedrockResp BedrockResponse
	if err := json.Unmarshal(resp.Body, &bedrockResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Convert to AnthropicResponse format
	return &AnthropicResponse{
		ID:      bedrockResp.ID,
		Type:    bedrockResp.Type,
		Role:    bedrockResp.Role,
		Content: bedrockResp.Content,
		Model:   c.model,
		Usage: Usage{
			InputTokens:  bedrockResp.Usage.InputTokens,
			OutputTokens: bedrockResp.Usage.OutputTokens,
		},
	}, nil
}

// ParseContent parses the response content - same implementation as AnthropicClient
func (c *BedrockClient) ParseContent(content []json.RawMessage) (string, []ToolUseContent, error) {
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