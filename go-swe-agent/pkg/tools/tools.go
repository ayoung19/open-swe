package tools

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type ToolExecutor struct {
	workingDir string
}

func NewToolExecutor(workingDir string) *ToolExecutor {
	return &ToolExecutor{
		workingDir: workingDir,
	}
}

func (t *ToolExecutor) Execute(name string, args map[string]interface{}) (string, error) {
	switch name {
	case "bash":
		return t.executeBash(args)
	case "read_file":
		return t.readFile(args)
	case "write_file":
		return t.writeFile(args)
	case "list_files":
		return t.listFiles(args)
	case "search":
		return t.search(args)
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

func (t *ToolExecutor) executeBash(args map[string]interface{}) (string, error) {
	command, ok := args["command"].(string)
	if !ok {
		return "", fmt.Errorf("bash requires 'command' parameter")
	}

	cmd := exec.Command("bash", "-c", command)
	cmd.Dir = t.workingDir
	
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	err := cmd.Run()
	
	output := stdout.String()
	if stderr.Len() > 0 {
		output += "\nSTDERR:\n" + stderr.String()
	}
	
	if err != nil && output == "" {
		return "", fmt.Errorf("command failed: %w", err)
	}
	
	return output, nil
}

func (t *ToolExecutor) readFile(args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("read_file requires 'path' parameter")
	}

	if !filepath.IsAbs(path) {
		path = filepath.Join(t.workingDir, path)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	return string(content), nil
}

func (t *ToolExecutor) writeFile(args map[string]interface{}) (string, error) {
	path, ok := args["path"].(string)
	if !ok {
		return "", fmt.Errorf("write_file requires 'path' parameter")
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("write_file requires 'content' parameter")
	}

	if !filepath.IsAbs(path) {
		path = filepath.Join(t.workingDir, path)
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	return fmt.Sprintf("File written successfully to %s", path), nil
}

func (t *ToolExecutor) listFiles(args map[string]interface{}) (string, error) {
	path := t.workingDir
	if p, ok := args["path"].(string); ok {
		if filepath.IsAbs(p) {
			path = p
		} else {
			path = filepath.Join(t.workingDir, p)
		}
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("failed to list directory: %w", err)
	}

	var result strings.Builder
	for _, entry := range entries {
		if entry.IsDir() {
			result.WriteString(fmt.Sprintf("[DIR]  %s\n", entry.Name()))
		} else {
			info, _ := entry.Info()
			size := int64(0)
			if info != nil {
				size = info.Size()
			}
			result.WriteString(fmt.Sprintf("[FILE] %s (%d bytes)\n", entry.Name(), size))
		}
	}

	return result.String(), nil
}

func (t *ToolExecutor) search(args map[string]interface{}) (string, error) {
	pattern, ok := args["pattern"].(string)
	if !ok {
		return "", fmt.Errorf("search requires 'pattern' parameter")
	}

	path := t.workingDir
	if p, ok := args["path"].(string); ok {
		if filepath.IsAbs(p) {
			path = p
		} else {
			path = filepath.Join(t.workingDir, p)
		}
	}

	// Use ripgrep if available, otherwise fall back to grep
	cmd := exec.Command("rg", "--no-heading", "--line-number", pattern, path)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		// Try grep as fallback
		cmd = exec.Command("grep", "-r", "-n", pattern, path)
		output, err = cmd.CombinedOutput()
		if err != nil && len(output) == 0 {
			return "No matches found", nil
		}
	}

	return string(output), nil
}

func GetAvailableTools() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"name":        "bash",
			"description": "Execute bash commands in the working directory",
			"input_schema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"command": map[string]interface{}{
						"type":        "string",
						"description": "The bash command to execute",
					},
				},
				"required": []string{"command"},
			},
		},
		{
			"name":        "read_file",
			"description": "Read the contents of a file",
			"input_schema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "The path to the file to read",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			"name":        "write_file",
			"description": "Write content to a file",
			"input_schema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "The path to the file to write",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "The content to write to the file",
					},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			"name":        "list_files",
			"description": "List files and directories in a given path",
			"input_schema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "The directory path to list (optional, defaults to working directory)",
					},
				},
			},
		},
		{
			"name":        "search",
			"description": "Search for a pattern in files using grep/ripgrep",
			"input_schema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "The pattern to search for",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "The path to search in (optional, defaults to working directory)",
					},
				},
				"required": []string{"pattern"},
			},
		},
	}
}