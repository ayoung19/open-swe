package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type BedrockClient struct {
	client *bedrockruntime.Client
	model  string
	region string
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

func (c *BedrockClient) CreateMessage(messages []AnthropicMessage, system string, tools []Tool) (*AnthropicResponse, error) {
	// Convert our messages to Bedrock format
	var converseMessages []types.Message
	
	for _, msg := range messages {
		bedrockMsg := types.Message{
			Role: types.ConversationRole(msg.Role),
		}

		// Handle different content types
		switch content := msg.Content.(type) {
		case []interface{}:
			var msgContent []types.ContentBlock
			for _, item := range content {
				switch v := item.(type) {
				case TextContent:
					msgContent = append(msgContent, &types.ContentBlockMemberText{
						Value: v.Text,
					})
				case ToolResultContent:
					msgContent = append(msgContent, &types.ContentBlockMemberToolResult{
						Value: types.ToolResultBlock{
							ToolUseId: aws.String(v.ToolUseID),
							Content: []types.ToolResultContentBlock{
								&types.ToolResultContentBlockMemberText{
									Value: types.ToolResultContentBlockText{
										Text: aws.String(v.Content),
									},
								},
							},
							Status: func() types.ToolResultStatus {
								if v.IsError {
									return types.ToolResultStatusError
								}
								return types.ToolResultStatusSuccess
							}(),
						},
					})
				}
			}
			bedrockMsg.Content = msgContent
		case []json.RawMessage:
			// This is from assistant responses
			var msgContent []types.ContentBlock
			text, toolCalls, _ := c.ParseContent(content)
			
			if text != "" {
				msgContent = append(msgContent, &types.ContentBlockMemberText{
					Value: text,
				})
			}
			
			for _, toolCall := range toolCalls {
				inputJSON, _ := json.Marshal(toolCall.Input)
				msgContent = append(msgContent, &types.ContentBlockMemberToolUse{
					Value: types.ToolUseBlock{
						ToolUseId: aws.String(toolCall.ID),
						Name:      aws.String(toolCall.Name),
						Input:     aws.JSONValue(inputJSON),
					},
				})
			}
			bedrockMsg.Content = msgContent
		}

		if len(bedrockMsg.Content) > 0 {
			converseMessages = append(converseMessages, bedrockMsg)
		}
	}

	// Convert tools to Bedrock format
	var bedrockTools []types.Tool
	if len(tools) > 0 {
		for _, tool := range tools {
			inputSchema, _ := json.Marshal(tool.InputSchema)
			bedrockTools = append(bedrockTools, &types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String(tool.Name),
					Description: aws.String(tool.Description),
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: aws.JSONValue(inputSchema),
					},
				},
			})
		}
	}

	// Build the request
	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(c.model),
		Messages: converseMessages,
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens: aws.Int32(8192),
		},
	}

	if system != "" {
		input.System = []types.SystemContentBlock{
			&types.SystemContentBlockMemberText{
				Value: system,
			},
		}
	}

	if len(bedrockTools) > 0 {
		input.ToolConfig = &types.ToolConfiguration{
			Tools: bedrockTools,
		}
	}

	// Make the API call
	resp, err := c.client.Converse(context.TODO(), input)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse error: %w", err)
	}

	// Convert response back to our format
	var responseContent []json.RawMessage
	
	for _, content := range resp.Output.Message.Content {
		switch v := content.(type) {
		case *types.ContentBlockMemberText:
			textContent := map[string]interface{}{
				"type": "text",
				"text": v.Value,
			}
			raw, _ := json.Marshal(textContent)
			responseContent = append(responseContent, raw)
			
		case *types.ContentBlockMemberToolUse:
			var inputMap map[string]interface{}
			if v.Value.Input != nil {
				json.Unmarshal(v.Value.Input, &inputMap)
			}
			
			toolContent := ToolUseContent{
				Type:  "tool_use",
				ID:    aws.ToString(v.Value.ToolUseId),
				Name:  aws.ToString(v.Value.Name),
				Input: inputMap,
			}
			raw, _ := json.Marshal(toolContent)
			responseContent = append(responseContent, raw)
		}
	}

	return &AnthropicResponse{
		ID:      "bedrock-" + aws.ToString(resp.ResponseMetadata.RequestID),
		Type:    "message",
		Role:    string(resp.Output.Message.Role),
		Content: responseContent,
		Model:   c.model,
		Usage: Usage{
			InputTokens:  int(aws.ToInt32(resp.Usage.InputTokens)),
			OutputTokens: int(aws.ToInt32(resp.Usage.OutputTokens)),
		},
	}, nil
}

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