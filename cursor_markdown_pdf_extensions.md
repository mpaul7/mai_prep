# Cursor IDE Extensions for Markdown to PDF Conversion

Since Cursor IDE is based on VS Code, it supports VS Code extensions. Here are the best options for converting Markdown to PDF:

## Method 1: Standard VS Code Extensions (Easiest)

### Option A: "Markdown PDF" by yzane

**Installation:**
1. Open Cursor IDE
2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac) to open Extensions
3. Search for "Markdown PDF" by yzane
4. Click Install

**Usage:**
1. Open your Markdown file
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open Command Palette
3. Type "Markdown PDF: Export (pdf)" and select it
4. The PDF will be generated in the same directory as your Markdown file

**Features:**
- Syntax highlighting
- Custom CSS styling
- Table of contents
- Page breaks
- Custom headers/footers

**Configuration:**
You can customize settings in Cursor Settings (JSON):
```json
{
  "markdown-pdf.styles": ["path/to/custom.css"],
  "markdown-pdf.includeDefaultStyles": true,
  "markdown-pdf.margin.top": "1cm",
  "markdown-pdf.margin.bottom": "1cm",
  "markdown-pdf.margin.right": "1cm",
  "markdown-pdf.margin.left": "1cm"
}
```

### Option B: "Markdown Preview Enhanced" by Yiyi Wang

**Installation:**
1. Open Extensions (`Ctrl+Shift+X`)
2. Search for "Markdown Preview Enhanced"
3. Click Install

**Usage:**
1. Open your Markdown file
2. Right-click in the editor
3. Select "Markdown Preview Enhanced: Export (pdf)"

**Features:**
- Live preview
- Math equations support
- Diagrams (Mermaid, PlantUML)
- Presentation mode
- More advanced features than Markdown PDF

## Method 2: MCP Server (Advanced)

### Markdown to PDF MCP Server

This is a Model Context Protocol server that integrates with Cursor's AI features.

**Installation:**
1. Open Cursor Settings
2. Navigate to `Features > MCP`
3. Click "+ Add New MCP Server"
4. Configure:
   - **Name:** Markdown to PDF
   - **Type:** stdio
   - **Command:** `node ~/path/to/markdown-to-pdf-server/build/index.js`

**Note:** This requires setting up the MCP server separately, which is more complex than standard extensions.

## Recommended: Use "Markdown PDF" Extension

For most users, the **"Markdown PDF" by yzane** extension is the easiest and most reliable option:

1. **Install:** Search for "Markdown PDF" in Extensions
2. **Use:** `Ctrl+Shift+P` → "Markdown PDF: Export (pdf)"
3. **Done!** PDF appears next to your Markdown file

## Quick Comparison

| Extension | Ease of Use | Features | Recommended For |
|-----------|-------------|----------|-----------------|
| Markdown PDF (yzane) | ⭐⭐⭐⭐⭐ | Basic to Advanced | Most users |
| Markdown Preview Enhanced | ⭐⭐⭐⭐ | Advanced | Users needing diagrams/math |
| MCP Server | ⭐⭐ | AI Integration | Advanced users |

## Troubleshooting

### Extension not found
- Make sure you're searching in the Extensions marketplace
- Cursor uses the VS Code extension marketplace, so all VS Code extensions should work

### PDF generation fails
- Check that Node.js is installed (required for most PDF extensions)
- Try restarting Cursor after installation
- Check the Output panel for error messages

### Styling issues
- Customize CSS in settings
- Use the extension's configuration options
- Check the extension's GitHub page for examples



