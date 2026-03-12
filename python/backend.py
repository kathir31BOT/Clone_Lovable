import os
import requests
import json
import ast
import sys
import io
import subprocess
import threading
import time
import fnmatch
import re
from pathlib import Path
from datetime import datetime
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from contextlib import redirect_stdout
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Server mode: when True, no local file writes are performed.
# All file operations are returned as messages to the extension.
# ------------------------------------------------------------
SERVER_MODE = True

# ------------------------------------------------------------
# Gemini API setup
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not configured")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-pro-preview"  # or whatever model you use

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="VibeCoding AI Backend")

# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    conversation_history: str = ""
    pending_action: Optional[dict] = None
    files: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    messages: List[dict]


SYSTEM_PROMPT = """
ROLE:
You are the Worktual VS Code Extension AI Assistant - an expert-level AI software architect and senior developer.
You act as BOTH:
1) a developer assistant explaining advanced concepts with production-ready examples
2) an execution agent that issues structured JSON actions for the extension.

3) website Builder: You can create complete websites with modern frameworks like React, Next.js, Vue, Angular, Svelte, etc. You generate all necessary files and configurations.

website building Rules:
- When user asks to create a website, you generate ALL necessary files (HTML, CSS, JS, config files) in ONE response. Do NOT create files one by one.
- Use modern frameworks and best practices for the requested website type.
- Always include proper error handling, security considerations, and optimizations in the generated code.

- The generated website must be have all the necessary files to run immediately after creation (no placeholders, no missing dependencies).

Your primary goal is to help users build, modify, debug, and run projects with PRODUCTION-QUALITY, ADVANCED-LEVEL code only.

If asked about yourself, reply:
"I am an AI assistant integrated into VS Code through the Worktual extension. I provide advanced, production-quality code and help create files, debug code, run programs, and manage projects using structured actions."

--------------------------------------------------
CRITICAL RULE: CREATING NEW FILES (MANDATORY)
--------------------------------------------------

When the user asks to CREATE, WRITE, or BUILD something NEW:

DETECT NEW CREATION BY KEYWORDS - If user says ANY of these, CREATE NEW files immediately:
- "write code for [anything]" - This ALWAYS means create NEW file
- "write a [filename.py/js/html]"
- "create a [something]"
- "build a [something]"
- "make a [something]"
- "implement [something]"
- "add two numbers" (simple math programs)
- "write hello world"
- "create hello world"
- Any request to write code that doesn't mention existing files

EXAMPLES - Just create the files directly:
- "write code for add two number" → create_file with add function
- "write a Python script for factorial" → create_file with the script
- "create a React component for login" → create necessary files
- "build a simple calculator" → create project files
- "write hello world in Python" → create file with hello world

ABSOLUTE RULE:
- "write code for [anything]" = CREATE NEW, never search for existing
- Don't ask which project - just create new files in current workspace
- For simple programs, no TODO.md needed - just create the file directly


--------------------------------------------------
CRITICAL RULE: SCALE CODE TO COMPLEXITY (MANDATORY)
--------------------------------------------------

SIMPLE REQUEST — single file, minimal code:
Keywords: "write", "add", "sum", "hello world", "factorial",
          "calculator", "sort", "reverse", "fibonacci"
Rules:
- Create ONE file only
- Write MINIMAL clean code — 10 to 30 lines MAX
- No extra comments, no unnecessary classes
- Just working code, nothing more

Example:
"write sum of two numbers" → 5 lines MAX
{"action": "create_file", "path": "sum.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(3, 5))"}

--------------------------------------------------

COMPLEX REQUEST — full project, ALL files:
Keywords: "crm", "erp", "ecommerce", "saas", "platform",
          "management system", "full stack", "complete project",
          "with backend", "with database", "with auth",
          "inventory", "hospital", "school", "booking system"
Rules:
- Generate EVERY file — frontend, backend, database, config
- NEVER stop at 200 lines — generate everything needed
- MUST include ALL of these:
  → Frontend (HTML/CSS/JS or React/Next.js with ALL pages)
  → Backend (FastAPI/Express/Django with ALL routes)
  → Database models (ALL tables/collections)
  → Authentication (JWT login, register, logout)
  → CRUD operations for ALL entities
  → requirements.txt or package.json
  → .env.example with all variables
  → README.md with setup instructions
- Every file must be COMPLETE — zero placeholders
- Code must run immediately after creation
- NEVER say "due to length I will simplify"
- NEVER truncate with "..." or "# rest of code here"
- NEVER generate partial files — always complete

--------------------------------------------------



--------------------------------------------------
CRITICAL RULE: SCALE CODE TO COMPLEXITY (MANDATORY)
--------------------------------------------------

You MUST detect request complexity and scale output accordingly:

SIMPLE REQUEST — single file, minimal code:
Keywords: "write", "add", "sum", "hello world", "factorial",
          "calculator", "sort", "reverse", "fibonacci"
Rules:
- Create ONE file only
- Write MINIMAL, clean code — 10 to 30 lines MAX
- No classes unless necessary
- No extra comments, no docstrings
- Just working code, nothing more

Example:
"write sum of two numbers"
→ 5 lines of code. No more.
{"action": "create_file", "path": "sum.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(3, 5))"}

MEDIUM REQUEST — small project, few files:
Keywords: "todo app", "weather app", "chat app",
          "login page", "dashboard", "simple api"
Rules:
- 3 to 6 files
- Each file 50 to 150 lines
- Include basic frontend + backend
- Include requirements.txt or package.json

COMPLEX REQUEST — full project, many files:
Keywords: "crm", "erp", "ecommerce", "saas", "platform",
          "management system", "full stack", "complete project",
          "with backend", "with database", "with auth"
Rules:
- Generate ALL files — frontend, backend, database, config
- NEVER stop at 200 lines — generate as much as needed
- MUST include:
  → Frontend (HTML/CSS/JS or React/Next.js)
  → Backend (FastAPI/Express/Django)
  → Database models (SQLAlchemy/Prisma/Mongoose)
  → Authentication (JWT/sessions)
  → API routes (CRUD operations)
  → requirements.txt or package.json
  → .env.example
  → README.md with setup instructions
- Each file must be COMPLETE — no placeholders, no "add logic here"
- Code must run immediately after creation

ABSOLUTE RULE:
- Simple request = simple code. NEVER over-engineer.
- Complex request = COMPLETE project. NEVER under-deliver.
- NEVER truncate code with "..." or "# rest of code here"
- NEVER say "due to length I will simplify" — generate everything

--------------------------------------------------
MANDATORY: ALWAYS OUTPUT JSON ACTION (CRITICAL)
--------------------------------------------------

When you create code, you MUST output a JSON action block. WITHOUT EXCEPTION.

SINGLE FILE — use create_file:
{"action": "create_file", "path": "filename.py", "content": "YOUR CODE HERE"}

MULTIPLE FILES (project) — use create_project with folder name + all files:
{"action": "create_project", "folder": "project_name", "files": [{"path": "main.py", "content": "..."}, {"path": "requirements.txt", "content": "..."}]}

CRITICAL RULE — PROJECTS WITH MULTIPLE FILES:
- When creating ANY project (CRM, API, app, website) with 2+ files, you MUST use create_project
- NEVER emit multiple separate create_file actions for a multi-file project
- The "folder" key is the project directory name (e.g. "crm_project", "my_app")
- The "files" array contains objects with "path" (filename only, no folder prefix) and "content"
- ALL files for the project go in the single create_project JSON — no separate create_file calls

Example — single file:
{"action": "create_file", "path": "add_numbers.py", "content": "def add(a, b):\n    return a + b"}

Example — multi-file project:
{"action": "create_project", "folder": "crm_app", "files": [{"path": "requirements.txt", "content": "fastapi\n"}, {"path": "main.py", "content": "from fastapi import FastAPI\napp = FastAPI()"}]}

CRITICAL OUTPUT FORMAT — INTERLEAVE TEXT AND JSON:
- Write a brief intro sentence (e.g. "Here's your calculator:") THEN immediately output the JSON action
- After the JSON, write a short explanation of what was created and how to run it
- NEVER put all JSON at the end — each file's JSON must appear right after mentioning that file
- The "content" field must contain the EXACT code (no markdown, no backticks)
- Do NOT wrap the content in backticks — plain text only
- For single files, path should be a simple filename like "add.py"
- For projects, path inside files[] should be the filename only (e.g. "main.py", not "crm/main.py")

CORRECT FORMAT EXAMPLE:
Here is your Python calculator:
{"action": "create_file", "path": "calc.py", "content": "..."}
Run it with: python calc.py. It supports +, -, *, / operations.

ABSOLUTE RULE: NEVER ASK FOR CONFIRMATION
- NEVER ask "Would you like me to create this file?"
- Just output the JSON action — the extension handles file creation automatically

--------------------------------------------------
CRITICAL RULE: MODIFYING EXISTING PROJECTS/FILES (MANDATORY WORKFLOW)
--------------------------------------------------

When the user asks to UPDATE or MODIFY an EXISTING project (like "update the UI", "fix this file", "add feature to existing code", "update CRM frontend", "improve the design"):

YOU MUST FOLLOW THIS EXACT WORKFLOW - NO EXCEPTIONS:

STEP 1 - SEARCH FOR EXISTING PROJECTS BY KEYWORD (MANDATORY):
- User provides keywords like "crm", "frontend", "backend", "api" - NOT exact folder names
- Use search_folders action with the keyword to find ALL matching projects
- Example: User says "update crm frontend" → search for folders containing "crm"
- Search patterns: *crm*, crm*, *crm (case-insensitive)
- LIST ALL MATCHING PROJECTS found:
  "I found these projects matching 'crm':
  1. /workspace/crm-project
  2. /workspace/crm-system
  3. /workspace/CRM-Backend
  Which project would you like to update?"

STEP 2 - WAIT FOR USER SELECTION (MANDATORY):
- NEVER assume which project to use
- WAIT for user to specify the exact project number or name
- Only proceed after user confirms which project

STEP 3 - SEARCH FOR EXISTING FILES IN SELECTED PROJECT (MANDATORY):
- Once user selects the project, use search_files with RECURSIVE patterns
- Search for ALL frontend files in the selected project:
  * **/*.html (in templates/, static/, src/, any subfolder)
  * **/*.css (in static/css/, styles/, assets/, any subfolder)
  * **/*.js (in static/js/, scripts/, src/, any subfolder)
- ALSO search in COMMON LOCATIONS:
  * templates/ folder
  * static/ folder and all subfolders
  * src/ folder
  * assets/ folder
  * public/ folder
  * Any other subdirectories
- LIST EVERY FILE FOUND with EXACT PATHS:
  "I found these existing files in /workspace/crm-project:
  - templates/index.html
  - templates/dashboard.html
  - static/css/style.css
  - static/css/dashboard.css
  - static/js/app.js
  - static/js/charts.js"

STEP 4 - GET USER CONFIRMATION (MANDATORY):
- EXPLICITLY ask: "I found these existing files. Do you want me to update these files?"
- WAIT for user to confirm YES before proceeding
- NEVER proceed without explicit user confirmation

STEP 5 - READ AND UPDATE EXISTING FILES (MANDATORY):
- For EACH existing file:
  a) Use get_file_info to read the complete file content from its CURRENT location
  b) Show the user the EXACT changes using SEARCH/REPLACE format
  c) ASK: "Do you want me to apply these changes to [exact file path]?"
  d) Only after user confirms, use update_file action with COMPLETE modified content
  e) Use the EXACT SAME FILE PATH - do NOT create new folders
  f) Move to next file only after current file is confirmed

STEP 6 - VERIFY AND SUMMARIZE:
- After all files are updated, list all modified files with their EXACT paths
- Confirm: "Successfully updated these existing files: [list of files]"

--------------------------------------------------
ABSOLUTE PROHIBITIONS - NEVER DO THESE:
--------------------------------------------------

1. NEVER create new folders when updating existing projects
2. NEVER create new files when user asks to UPDATE existing ones
3. NEVER move files from their current locations
4. NEVER create folders like "crm_frontend" when "crm" project already exists
5. NEVER assume which project to modify - always search and ask
6. NEVER proceed without explicit user confirmation at each step
7. NEVER search only root directory - always search recursively

--------------------------------------------------
EXAMPLE CORRECT WORKFLOW:
--------------------------------------------------

User: "Update the crm frontend"

AI: [Uses search_folders with keyword "crm"]
"I found these projects matching 'crm':
1. /workspace/crm-project
2. /workspace/crm-system-v2
3. /workspace/CRM-Backend
Which project would you like to update?"

User: "crm-project" (or "1")

AI: [Uses search_files with **/*.html, **/*.css, **/*.js in /workspace/crm-project]
"I found these existing files in /workspace/crm-project:
- templates/index.html
- templates/dashboard.html
- static/css/main.css
- static/css/dashboard.css
- static/js/app.js
- static/js/charts.js
Do you want me to update these existing files?"

User: "Yes"

AI: [Reads templates/index.html] "Here are the changes for templates/index.html:

SEARCH/REPLACE format:
<<<<<<< SEARCH
<div class="header">
  <h1>Old Title</h1>
</div>
=======
<div class="header modern">
  <h1>New Modern Title</h1>
  <nav class="navbar">...</nav>
</div>
>>>>>>> REPLACE

Do you want me to apply these changes to templates/index.html?"

User: "Yes"

AI: [Uses update_file action for templates/index.html with complete updated content]
[Continues with next files...]

--------------------------------------------------
CODE QUALITY STANDARDS (MANDATORY)
--------------------------------------------------

You MUST ALWAYS provide ADVANCED-LEVEL, PRODUCTION-READY code:

1. MODERN BEST PRACTICES:
   - Use latest language features and syntax (ES2023+, Python 3.11+, TypeScript 5.0+, etc.)
   - Implement proper error handling with try/catch, async/await patterns
   - Use type hints, interfaces, and strict typing where applicable
   - Follow SOLID principles and clean architecture patterns

2. PRODUCTION-READY STANDARDS:
   - Include comprehensive input validation and sanitization
   - Implement proper logging and monitoring hooks
   - Add security best practices (CSRF protection, XSS prevention, SQL injection prevention)
   - Write efficient, optimized algorithms with O(n) considerations
   - Include proper resource cleanup and memory management

3. ADVANCED PATTERNS:
   - Use design patterns (Factory, Singleton, Observer, Dependency Injection) appropriately
   - Implement proper abstraction layers and separation of concerns
   - Use reactive programming, functional programming concepts where beneficial
   - Include proper state management for complex applications

4. COMPLETE IMPLEMENTATIONS:
   - NEVER provide partial or placeholder code
   - ALWAYS include all necessary imports, dependencies, and configurations
   - Provide working examples that can run immediately
   - Include unit tests or test examples for critical functionality

5. DOCUMENTATION:
   - Add comprehensive JSDoc/docstring comments
   - Include README with setup instructions
   - Document API endpoints, function parameters, and return types
   - Add inline comments for complex logic

--------------------------------------------------
OUTPUT MODES (STRICT PRIORITY ORDER)
--------------------------------------------------

You MUST decide the mode before responding:

MODE 1 — ACTION MODE (filesystem or execution change)
Triggered when user requests:
- create/build/generate project
- create/update/delete files or folders
- fix/debug code
- run code / execute code / compile and run

Response structure:
1) Brief explanation (1–3 sentences max)
2) Required code previews (if creating/updating files) - MUST be production-ready
3) JSON action blocks

--------------------------------------------------
CRITICAL RULE: RUNNING / EXECUTING CODE
--------------------------------------------------

When the user says ANY of these, you MUST emit a run_file JSON action — NEVER just describe how to run it:
- "run [filename]"
- "execute [filename]"  
- "execute [filename] code"
- "compile and run [filename]"
- "run this"
- "execute this"
- "test this"
- "run the code"

You MUST output ONLY this JSON (no terminal instructions, no "here are the commands"):
{"action": "run_file", "path": "ExactFileName.java"}

EXAMPLE:
User: "execute ThreeSum.java code"
CORRECT output:
Running ThreeSum.java now.
{"action": "run_file", "path": "ThreeSum.java"}

WRONG output (NEVER DO THIS):
Here are the commands you need to run:
Compile: javac ThreeSum.java
Run: java ThreeSum

ABSOLUTE RULE: If the user says "run" or "execute", ALWAYS emit run_file JSON. NEVER give terminal instructions.

MODE 2 — EXPLANATION MODE
Triggered when user asks for:
- explanations
- examples
- concepts
- learning help

Response:
- Markdown text only with ADVANCED examples
- NO JSON
- Include production-ready code examples, never basic tutorials

--------------------------------------------------
CRITICAL FORMATTING RULES
--------------------------------------------------

1. All code shown to users MUST be inside markdown blocks:
   ```language
   production-ready code here
   ```

2. NEVER provide:
   - Basic "hello world" examples
   - Placeholder comments like "// add your logic here"
   - Incomplete implementations
   - Deprecated or outdated patterns
   - Code without proper error handling

3. ALWAYS provide:
   - Full working implementations
   - Modern syntax and patterns
   - Proper error handling and edge case coverage
   - Security considerations
   - Performance optimizations
   - Type safety (TypeScript types, Python type hints, etc.)

4. When modifying existing files:
   - Show the exact SEARCH/REPLACE blocks
   - Ask for confirmation before applying
   - - Never overwrite entire files unless explicitly requested


--------------------------------------------------
CRITICAL RULE: CODE OUTPUT SCALE (MANDATORY)
--------------------------------------------------

Before writing any code, you MUST analyze the user's request and ask yourself:
"How big and complex does this project need to be to actually work?"
Then generate EXACTLY that much code — no more, no less.

JUDGE complexity from the request itself:

If the request is a SINGLE UTILITY — a function, script, or small tool:
→ Write minimal clean code, 10-30 lines, one file, no over-engineering

If the request is a SMALL APP — a page, component, or simple tool with UI:
→ Write 3-5 files, 50-150 lines each, working frontend + logic

If the request is a FULL PROJECT — anything with users, data, dashboards,
multiple pages, login, database, API, or any real-world application:
→ You MUST generate a COMPLETE, FULLY WORKING project
→ Every single file must be production-ready and runnable
→ Do NOT stop until every feature mentioned by the user is implemented

FOR ANY FULL PROJECT — these are non-negotiable minimums:

BACKEND — must have ALL of:
- Every API route the project needs (GET, POST, PUT, DELETE)
- Real database queries — no fake/hardcoded data
- Input validation and error handling on every route
- JWT authentication with login, register, logout
- Minimum 300 lines

FRONTEND HTML pages — must have ALL of:
- Full working navigation
- Real data loaded from the API
- Forms that actually submit to the backend
- Tables/lists that show real data with search and filter
- Modals for create, edit, delete
- Success and error feedback to user
- Minimum 400 lines per page

CSS — must have:
- Professional modern design
- Fully responsive (mobile + desktop)
- Minimum 200 lines

JAVASCRIPT — must have:
- Every API call wired up with fetch/async/await
- JWT token handling (store, send in headers, logout)
- Form validation before submit
- Dynamic rendering of data from API
- Minimum 300 lines

DATABASE MODELS — must have:
- Every table the project needs
- All relationships and foreign keys
- Timestamps on every model
- Minimum 150 lines

ALSO ALWAYS INCLUDE:
- requirements.txt or package.json with all dependencies
- .env.example with every variable the project needs
- README.md explaining how to run the project

ABSOLUTE RULES — NEVER BREAK THESE:
- Read the user's request carefully — build exactly what they asked for
- If user says "with auth" → implement full JWT auth
- If user says "with dashboard" → build a real dashboard with real data
- If user says "with reports" → build working report pages
- NEVER truncate code — always complete every file fully
- NEVER write "# rest of the code here" or "// TODO"
- NEVER write empty functions or placeholder logic
- NEVER say "due to length I'll simplify" — generate everything
- ALL files must be in ONE create_project JSON block, fully closed with ]}

   
--------------------------------------------------
CRITICAL RULE: GIT / GITHUB OPERATIONS — DO NOT HANDLE
--------------------------------------------------

NEVER generate any scripts, code, or files for git operations.
This includes: git push, git commit, git init, pushing to GitHub,
creating push_to_github.py, deploy scripts, or ANY git-related code.

When user says ANY of these:
- "push to git"
- "push to github"
- "push my code"
- "push this to github"
- "commit and push"
- "upload to github"
- ANY variation of pushing/uploading code to git/github

You MUST respond with ONLY this exact text:
"🔄 Starting git push flow..."

NOTHING ELSE. No scripts. No JSON actions. No code.
The extension handles all git operations internally.


### Why This Fixes Everything:
```
BEFORE:
User: "push to git"
         ↓
GitPushHandler.isGitPushIntent() 
         ↓ (even if this runs first)
Gemini also sees the message
         ↓
Gemini generates push_to_github.py ← problem

AFTER:
User: "push to git"
         ↓
GitPushHandler.isGitPushIntent() → runs our flow
         ↓
Even IF it reaches Gemini → 
Gemini now knows to say ONLY
"🔄 Starting git push flow..."
→ No script generated ✅


NEVER put all files inside one giant create_project JSON.
This causes JSON parse failures for large projects.

Instead, output EACH FILE as a SEPARATE create_file action:

{"action": "create_file", "path": "project_name/main.py", "content": "...full code..."}
{"action": "create_file", "path": "project_name/auth.py", "content": "...full code..."}
{"action": "create_file", "path": "project_name/models.py", "content": "...full code..."}
{"action": "create_file", "path": "project_name/static/index.html", "content": "...full code..."}

Each JSON is on its own line. Path includes the project folder name.
This way every file is parsed independently and written completely.


"""
# ------------------------------------------------------------
# Utility functions (unchanged, but used only for validation etc.)
# ------------------------------------------------------------
def format_file_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def validate_python_code(code, filename):
    try:
        ast.parse(code)
        return None, None
    except SyntaxError as e:
        lines = code.split('\n')
        error_line = lines[e.lineno - 1] if 0 < e.lineno <= len(lines) else ""
        pointer = " " * (e.offset - 1) + "^" if e.offset else ""
        suggestion = get_syntax_error_suggestion(e.msg, error_line)
        error_msg = (
            f"SyntaxError: {e.msg}\n"
            f"  File: {filename}\n"
            f"  Line: {e.lineno}\n"
            f"  Column: {e.offset}\n"
            f"  Code: {error_line.strip()}\n"
            f"        {pointer}"
        )
        if suggestion:
            error_msg += f"\n  Suggestion: {suggestion}"
        return error_msg, e.lineno
    except Exception as e:
        return f"Error: {str(e)}", None

def get_syntax_error_suggestion(error_msg, error_line):
    suggestions = {
        'invalid syntax': "Check for missing colons (:), brackets, or quotes",
        'unexpected EOF': "Check for unclosed brackets, quotes, or parentheses",
        'EOL while scanning string literal': "Check for unclosed quotes in strings",
        'unexpected indent': "Check indentation - Python uses consistent indentation",
        'unindent does not match': "Check that indentation levels match",
        'Missing parentheses': "Add missing parentheses ()",
        'invalid character': "Remove or replace invalid characters",
    }
    for key, suggestion in suggestions.items():
        if key.lower() in error_msg.lower():
            return suggestion
    if ':' not in error_line and any(keyword in error_line for keyword in ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with', 'elif', 'else']):
        return "Missing colon (:) at the end of the statement"
    if '(' in error_line and ')' not in error_line:
        return "Missing closing parenthesis )"
    if '[' in error_line and ']' not in error_line:
        return "Missing closing bracket ]"
    if '{' in error_line and '}' not in error_line:
        return "Missing closing brace }"
    return None

def analyze_error(error_msg, code, filename):
    # (same as original, returns dict)
    analysis = {
        'error_type': None,
        'error_message': error_msg,
        'line_number': None,
        'suggestions': [],
        'common_causes': [],
        'fix_examples': []
    }
    # Determine error type
    if 'SyntaxError' in error_msg:
        analysis['error_type'] = 'Syntax Error'
        analysis['common_causes'] = [
            'Missing colons (:) after control statements',
            'Unclosed brackets, parentheses, or quotes',
            'Incorrect indentation',
            'Invalid characters or typos'
        ]
    elif 'IndentationError' in error_msg:
        analysis['error_type'] = 'Indentation Error'
        analysis['common_causes'] = [
            'Mixed tabs and spaces',
            'Incorrect indentation level',
            'Missing indentation in block'
        ]
    elif 'NameError' in error_msg:
        analysis['error_type'] = 'Name Error'
        analysis['common_causes'] = [
            'Variable not defined',
            'Typo in variable name',
            'Variable defined in different scope',
            'Missing import statement'
        ]
    elif 'TypeError' in error_msg:
        analysis['error_type'] = 'Type Error'
        analysis['common_causes'] = [
            'Operating on incompatible types',
            'Wrong number of arguments',
            'NoneType operations',
            'String/number concatenation'
        ]
    elif 'IndexError' in error_msg or 'KeyError' in error_msg:
        analysis['error_type'] = 'Index/Key Error'
        analysis['common_causes'] = [
            'Accessing index out of range',
            'Key not found in dictionary',
            'Empty list/dict access',
            'Off-by-one errors'
        ]
    elif 'AttributeError' in error_msg:
        analysis['error_type'] = 'Attribute Error'
        analysis['common_causes'] = [
            'Method/property doesn\'t exist on object',
            'NoneType attribute access',
            'Wrong object type',
            'Missing import or module'
        ]
    elif 'ImportError' in error_msg or 'ModuleNotFoundError' in error_msg:
        analysis['error_type'] = 'Import Error'
        analysis['common_causes'] = [
            'Module not installed',
            'Incorrect module name',
            'Circular import',
            'Module not in PYTHONPATH'
        ]
    elif 'ZeroDivisionError' in error_msg:
        analysis['error_type'] = 'Zero Division Error'
        analysis['common_causes'] = [
            'Division by zero',
            'Modulo by zero',
            'Uninitialized denominator'
        ]
    elif 'FileNotFoundError' in error_msg:
        analysis['error_type'] = 'File Not Found Error'
        analysis['common_causes'] = [
            'File doesn\'t exist at path',
            'Wrong file path',
            'Permission denied',
            'Relative path issues'
        ]
    else:
        analysis['error_type'] = 'Runtime Error'
        analysis['common_causes'] = [
            'Logic error in code',
            'Unexpected input data',
            'Resource not available',
            'External dependency failure'
        ]
    
    # Extract line number if present
    import re
    line_match = re.search(r'line (\d+)', error_msg, re.IGNORECASE)
    if line_match:
        analysis['line_number'] = int(line_match.group(1))
    
    # Generate suggestions based on error type
    if analysis['line_number'] and code:
        lines = code.split('\n')
        if 0 < analysis['line_number'] <= len(lines):
            error_line = lines[analysis['line_number'] - 1]
            analysis['error_line'] = error_line.strip()
            
            # Add specific suggestions based on line content
            if analysis['error_type'] == 'Syntax Error':
                if ':' not in error_line and any(kw in error_line for kw in ['if', 'for', 'while', 'def', 'class']):
                    analysis['suggestions'].append("Add a colon (:) at the end of the line")
                if '(' in error_line and ')' not in error_line:
                    analysis['suggestions'].append("Add missing closing parenthesis )")
    
    return analysis

def format_error_analysis(analysis):
    lines = [f"[ERROR ANALYSIS] {analysis['error_type']}"]
    lines.append("-" * 50)
    if analysis['line_number']:
        lines.append(f"Location: Line {analysis['line_number']}")
        if 'error_line' in analysis:
            lines.append(f"Code: {analysis['error_line']}")
    lines.append(f"\nMessage: {analysis['error_message']}")
    if analysis['common_causes']:
        lines.append("\nCommon Causes:")
        for cause in analysis['common_causes']:
            lines.append(f"  • {cause}")
    if analysis['suggestions']:
        lines.append("\nSuggested Fixes:")
        for suggestion in analysis['suggestions']:
            lines.append(f"  → {suggestion}")
    return "\n".join(lines)

def execute_and_capture_errors(code):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(code, {"__name__": "__main__"})
        return None, stdout_capture.getvalue(), stderr_capture.getvalue()
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}", stdout_capture.getvalue(), stderr_capture.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _sanitize_json_string(s: str) -> str:
    """
    Fix common AI JSON generation mistakes before parsing:
    1. Unescaped literal newlines inside string values  -> \n
    2. Unescaped literal tabs inside string values      -> \t
    3. Unescaped literal carriage returns               -> \r
    This is the #1 reason json.loads fails on AI output.
    """
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            escape_next = False
            result.append(ch)
        elif ch == '\\':
            escape_next = True
            result.append(ch)
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string:
            # Inside a JSON string, newlines must be escaped
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def extract_json_objects(s):
    """
    Extract all top-level JSON objects from a string, returning (obj, start, end) tuples.
    Handles AI output that contains unescaped newlines/tabs inside string values.
    """
    if not isinstance(s, str):
        return []
    objects = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '{':
            start = i
            brace_depth = 0
            in_string = False
            escape_next = False
            while i < n:
                char = s[i]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            obj_str = s[start:i+1]
                            # First try raw parse
                            obj = None
                            try:
                                obj = json.loads(obj_str)
                            except json.JSONDecodeError:
                                # Retry after sanitizing unescaped control chars in strings
                                try:
                                    obj = json.loads(_sanitize_json_string(obj_str))
                                except json.JSONDecodeError:
                                    pass
                            if obj is not None:
                                objects.append((obj, start, i+1))
                            break
                i += 1
        i += 1
    return objects


def check_gemini_available():
    try:
        client.models.list()
        return True
    except:
        return False

# ------------------------------------------------------------
# Action handlers (server mode: return messages instead of executing)
# ------------------------------------------------------------
def create_folder_action(folder: str) -> dict:
    """Return a message to create a folder."""
    return {"type": "create_folder", "folder_path": folder}

def create_file_action(path: str, content: str) -> dict:
    """Return a message to create/update a file."""
    return {"type": "create_file", "file_path": path, "content": content}

def create_files_action(files: List[dict]) -> dict:
    """Return a message to create multiple files."""
    return {"type": "create_files", "files": files}

def ask_test_confirmation_action(path: str = None) -> dict:
    """Return a message asking user if they want to test the code."""
    return {"type": "confirmation", "text": "Can I test this code?", "action": {"intent": "test_code", "path": path}}

def status_message(text: str) -> dict:
    return {"type": "status", "text": text}

def error_message(text: str) -> dict:
    return {"type": "error", "text": text}

def response_message(text: str) -> dict:
    return {"type": "response", "text": text}

def confirmation_message(text: str, action: dict) -> dict:
    return {"type": "confirmation", "text": text, "action": action}

# For actions that require confirmation (file exists etc.), we generate a confirmation message.
def handle_create_file(path: str, content: str, confirmed: bool = False) -> List[dict]:
    """
    In server mode, we only simulate existence check.
    If not confirmed and file would exist, return confirmation message.
    Otherwise return create_file action.
    """
    # In server mode we cannot know if file exists on client side,
    # so we always just send the create_file action. The extension will handle duplicates.
    # For simplicity, we never ask for confirmation; the extension manages it.
    messages = [create_file_action(path, content)]
    # Add confirmation to ask if user wants to test the code (only for new code)
    messages.append(ask_test_confirmation_action(path))
    return messages

def handle_update_file(path: str, content: str, confirmed: bool = False) -> List[dict]:
    messages = [create_file_action(path, content)]  # same as create
    # For updates (error fixes), don't ask - just run the code
    # Let the extension handle auto-run after debug
    return messages
    
def handle_create_project(folder: str, files: List[dict]) -> List[dict]:
    # Return a single compound action
    return [{"type": "create_project", "folder": folder, "files": files}]

def handle_run_file(path: str, environment: str = "none") -> List[dict]:
    # Return a message that the extension will interpret to run the file.
    # The extension will open a terminal and execute.
    return [{"type": "run_file", "path": path, "environment": environment}]

def handle_debug_file(path: str, debug_stage: str = "all") -> List[dict]:

    """Return a message to trigger debugging on the extension side."""
    return [{"type": "debug_file", "path": path}]
# ------------------------------------------------------------
# AI processing
# ------------------------------------------------------------
COMPLEX_KEYWORDS = [
    'crm', 'erp', 'ecommerce', 'saas', 'platform', 'management system',
    'full stack', 'complete project', 'inventory', 'hospital', 'school',
    'booking', 'hotel', 'restaurant', 'clinic', 'real estate', 'hrm',
    'payroll', 'banking', 'finance', 'admin panel', 'pos system',
    'library system', 'attendance', 'leave management', 'ticket system'
]

COMPLEX_BOOSTER = """
==========================================================
MANDATORY FOR THIS REQUEST — FULL PROJECT GENERATION MODE
==========================================================

This is a COMPLEX FULL PROJECT. You MUST generate EVERY file completely.
Do NOT stop until all files are 100% written. Never truncate.

MINIMUM LINE REQUIREMENTS — HARD LIMITS:

index.html / login.html:
- 500+ lines
- Full page: navbar, hero section, login/register forms with validation
- CSS included inline or linked, JS for form submission to backend API
- Professional modern design with animations

dashboard.html:
- 500+ lines  
- Full sidebar navigation with all menu items
- Stats cards showing real data from API (total customers, revenue, etc.)
- Full data table with search bar, filter dropdowns, pagination controls
- Create/Edit/Delete modals with complete forms
- Loading spinners, success/error toast notifications

style.css:
- 300+ lines
- CSS variables for colors, fonts, spacing
- Full responsive design — mobile, tablet, desktop
- Sidebar styles, card styles, table styles, modal styles, button styles
- Hover effects and smooth transitions

app.js / dashboard.js:
- 300+ lines each
- All API calls using fetch with async/await
- JWT token stored in localStorage, sent in Authorization header
- Dynamic table rendering from API data
- Search and filter working in real time
- Create/Edit/Delete wired to backend API
- Form validation with inline error messages
- Loading states and error handling

auth.js:
- 200+ lines
- Login and register form submission
- JWT stored after login, redirect to dashboard
- Logout clears token
- Token validation on page load

main.py / app.py:
- 300+ lines
- ALL CRUD routes for every entity (GET all, GET one, POST, PUT, DELETE)
- JWT auth middleware protecting all routes
- Input validation using Pydantic schemas
- Proper error responses with status codes
- CORS configured, static files served

models.py:
- 200+ lines
- ALL database tables for the project
- Foreign key relationships between tables
- created_at, updated_at on every model
- Proper indexes

auth.py:
- 200+ lines
- JWT token creation with expiry
- Password hashing with bcrypt
- /register, /login, /logout, /me endpoints
- Token refresh endpoint
- Role-based access (admin/user)

schemas.py:
- 150+ lines
- Pydantic models for every request and response
- Input validation rules

ABSOLUTE RULES:
- Write every file from line 1 to the last line — NO shortcuts
- NEVER use "..." to skip code
- NEVER write "# rest of implementation here"  
- NEVER write empty functions
- ALL files in ONE create_project JSON block
- Close JSON with ]} before ending your response
==========================================================
"""

def process_message(user_input: str, conversation_history: str = "") -> str:
    """Call Gemini and return the raw text reply."""
    if not check_gemini_available():
        return "Error: Cannot connect to Gemini API. Please check your API key."

    greeting_keywords = ['hi', 'hello', 'hey', 'help', 'start']
    user_lower = user_input.lower().strip()
    if any(user_lower.startswith(kw) for kw in greeting_keywords) and len(user_input) < 20:
        return "Hello! Great to connect. What are we building today?"

    # Inject booster for complex project requests
    is_complex = any(kw in user_lower for kw in COMPLEX_KEYWORDS)
    booster = COMPLEX_BOOSTER if is_complex else ""

    try:
        full_prompt = f"{SYSTEM_PROMPT}{booster}\n\nConversation history:\n{conversation_history}\n\nUser: {user_input}\nAssistant:"
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config={"max_output_tokens": 32000, "temperature": 0.2}
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_user_message(request: ChatRequest) -> List[dict]:
    """
    Main entry point for /chat.
    Returns a list of messages (dicts) to be sent back to the extension.
    """
    messages = []

    # 1. If there is a pending action, handle it as a confirmation.
    if request.pending_action:
        action_data = request.pending_action
        act = action_data.get("action") or action_data.get("intent")
        if act == "create_file":
            path = action_data.get("path") or action_data.get("file_path")
            content = action_data.get("content", "")
            if path:
                messages.extend(handle_create_file(path, content, confirmed=True))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "update_file":
            path = action_data.get("path") or action_data.get("file_path")
            content = action_data.get("content", "")
            if path:
                messages.extend(handle_update_file(path, content, confirmed=True))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "create_folder":
            folder = action_data.get("folder") or action_data.get("folder_path")
            if folder:
                messages.append(create_folder_action(folder))
            else:
                messages.append(error_message("Missing folder in pending action"))
        elif act == "create_project":
            folder = action_data.get("folder") or action_data.get("project")
            files = action_data.get("files", [])
            if folder and files:
                messages.extend(handle_create_project(folder, files))
            else:
                messages.append(error_message("Missing folder or files in pending action"))
        elif act == "run_file":
            path = action_data.get("path") or action_data.get("file_path")
            env = action_data.get("environment", "none")
            if path:
                messages.extend(handle_run_file(path, env))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "test_code":
            path = action_data.get("path")
            if path:
                messages.extend(handle_run_file(path, "none"))
            else:
                # Try to find the main entry point
                messages.append({"type": "auto_debug"})
        else:
            messages.append(error_message(f"Unknown pending action: {act}"))
        return messages

    # 2. No pending action: process the user message normally.
    assistant_reply = process_message(request.message, request.conversation_history)

    # Extract JSON objects with positions
    json_items = extract_json_objects(assistant_reply)
    json_items.sort(key=lambda x: x[1])  # sort by start position

    # Build interleaved list of (text_chunk, json_obj) in original order.
    # This preserves the AI's natural structure:
    #   "Here's what I'll create:" → create_file → "And here's the explanation:" → create_file
    import re as _re
    FILE_ACTIONS = {"create_file","create file","createfile",
                    "create_project","create project","createproject",
                    "update_file","update file","updatefile"}

    interleaved = []   # list of ("text", str) or ("action", obj)
    last_end = 0
    for obj, start, end in json_items:
        # Text chunk before this JSON block
        chunk_start = start
        while chunk_start > last_end and assistant_reply[chunk_start - 1] in (' ', '\t', '\n', '\r'):
            chunk_start -= 1
        chunk = assistant_reply[last_end:chunk_start].strip()
        if chunk:
            interleaved.append(("text", chunk))
        interleaved.append(("action", obj))
        # Eat trailing whitespace after JSON
        chunk_end = end
        while chunk_end < len(assistant_reply) and assistant_reply[chunk_end] in (' ', '\t', '\n', '\r'):
            chunk_end += 1
        last_end = chunk_end
    # Remaining text after last JSON block
    tail = assistant_reply[last_end:].strip()
    if tail:
        interleaved.append(("text", tail))

    def clean_text_chunk(t):
        # Strip leaked JSON action blocks
        t = _re.sub(r'\{[^{}]*?"(?:action|intent)"[^{}]*?(?:\{[^{}]*?\}[^{}]*?)*\}', '', t, flags=_re.DOTALL)
        t = _re.sub(r'\{\s*"action".*', '', t, flags=_re.DOTALL)
        # Strip code fences (they duplicate file card content)
        t = _re.sub(r'```[\s\S]*?```', '', t)
        t = _re.sub(r'\n{3,}', '\n\n', t).strip()
        # Close any dangling fence
        if t.count('```') % 2 != 0:
            t += '\n```'
        return t

    # For backwards compat keep cleaned_text for the project-grouping logic below
    cleaned_text = clean_text_chunk(' '.join(
        t for kind, t in interleaved if kind == "text"
    ))

    # ── Group bare create_file actions into a create_project when appropriate ──
    # (response_message is appended AFTER file actions so cards appear before text)
    # When the AI sends N separate create_file actions all with bare filenames
    # (no directory separator) for what is clearly a project request, we wrap
    # them into a single create_project so all files land in one folder.
    bare_creates = [
        (obj, start, end) for obj, start, end in json_items
        if (obj.get("action") or "").strip().lower() in ("create_file", "create file", "createfile")
        and not (obj.get("path") or "").replace("\\", "/").strip("/").count("/")
    ]
    project_keywords = [
        "project", "crm", "app", "api", "website", "backend", "frontend",
        "system", "module", "service", "dashboard"
    ]
    user_lower_req = request.message.lower()
    looks_like_project = (
        len(bare_creates) >= 3 and
        any(kw in user_lower_req for kw in project_keywords)
    )

    if looks_like_project and len(bare_creates) == len([x for x in json_items if (x[0].get("action") or "").strip().lower() not in ("run_file","debug_file","create_folder","search_folders","search_files")]):
        # Derive folder name from user message
        import re as _re2
        folder_match = _re2.search(
            r"(?:create|build|make|generate|write)\s+(?:a\s+)?(?:crm|api|app|project|website|backend|frontend|system|service|dashboard)?\s*(?:project|app|system|backend|api|dashboard)?\s*(?:named|called|for)?\s*[\"']?([a-zA-Z0-9_\-]+)[\"']?",
            user_lower_req
        )
        # Simple fallback: grab first meaningful noun from request
        words = _re2.findall(r'[a-z][a-z0-9_]+', user_lower_req)
        skip = {"create","build","make","generate","write","a","an","the","with","and","for","in","of","to"}
        proj_words = [w for w in words if w not in skip and len(w) > 2]
        folder_name = "_".join(proj_words[:3]) if proj_words else "project"
        folder_name = folder_name.replace(" ", "_").replace("-", "_")[:40]

        grouped_files = [{"path": obj.get("path",""), "content": obj.get("content","")}
                         for obj, _, _ in bare_creates]
        messages.extend(handle_create_project(folder_name, grouped_files))
        # Skip the individual create_file actions below (remove them from json_items)
        bare_paths = {obj.get("path") for obj, _, _ in bare_creates}
        json_items = [(obj, s, e) for obj, s, e in json_items
                      if not ((obj.get("action") or "").strip().lower() in ("create_file","create file","createfile")
                              and obj.get("path") in bare_paths)]

    # Process each JSON object for actions (using the parsed objects)
    for obj, _, _ in json_items:   # <-- unpack tuple correctly
        action = obj.get("action") or obj.get("intent")
        if not action:
            continue
        act = action.strip().lower()

        if act in ("create_folder", "create folder", "createfolder"):
            folder = obj.get("folder") or obj.get("name")
            if folder:
                messages.append(create_folder_action(folder))
            else:
                messages.append(error_message("Missing folder name"))

        elif act in ("create_project", "create project", "createproject"):
            folder = obj.get("folder") or obj.get("name") or obj.get("project")
            files = obj.get("files", [])
            if folder and files:
                messages.extend(handle_create_project(folder, files))
            else:
                messages.append(error_message("Missing folder name or files list"))

        elif act in ("create_file", "create file", "createfile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            content = obj.get("content", "")
            if path:
                messages.extend(handle_create_file(path, content))
            else:
                messages.append(error_message("Missing path"))

        elif act in ("update_file", "update file", "updatefile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            content = obj.get("content", "")
            if path:
                messages.extend(handle_update_file(path, content))
            else:
                messages.append(error_message("Missing path"))

        elif act in ("run_file", "run file", "runfile", "test_file", "test file", "testfile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            env = obj.get("environment", "none")
            if path:
                messages.extend(handle_run_file(path, env))
            else:
                messages.append(error_message("Missing path"))

        elif act in ("debug_file", "debug file", "debugfile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            stage = obj.get("stage", "all")
            if path:
                messages.extend(handle_debug_file(path, stage))
            else:
                messages.append(error_message("Missing path"))
        elif act in ("auto_debug", "auto debug", "autodebug"):
            messages.append({"type": "auto_debug"})
        elif act in ("search_files", "search files", "searchfiles"):
            messages.append(status_message("File search is handled locally by the extension."))
        elif act in ("search_folders", "search folders", "searchfolders"):
            messages.append(status_message("Folder search is handled locally by the extension."))
        elif act in ("search_in_files", "search in files", "searchinfiles", "grep"):
            messages.append(status_message("Content search is handled locally by the extension."))
        elif act in ("get_file_info", "get file info", "getfileinfo", "file_info"):
            messages.append(status_message("File info is handled locally by the extension."))
        else:
            messages.append(error_message(f"Unknown action: {act}"))

    # ── Emit messages in interleaved order (text ↔ file actions) ──────────────
    # Build a map: action_obj → list of messages it produces
    # so we can insert each text chunk at its natural position.
    # We rebuild messages from scratch using interleaved order.
    #
    # NOTE: the project-grouping / action-processing loop above already appended
    # file action messages into `messages`. We need to rebuild in interleaved order,
    # so we clear and re-emit.
    messages_ordered = []

    # Re-process interleaved items in original order
    for kind, item in interleaved:
        if kind == "text":
            chunk = clean_text_chunk(item)
            if chunk:
                messages_ordered.append(response_message(chunk))
        else:
            # item is a json obj — emit its file action(s)
            obj = item
            action = (obj.get("action") or obj.get("intent") or "").strip().lower()
            if action in ("create_file", "create file", "createfile"):
                path = obj.get("path") or obj.get("filename") or obj.get("file")
                if path:
                    messages_ordered.extend(handle_create_file(path, obj.get("content", "")))
            elif action in ("update_file", "update file", "updatefile"):
                path = obj.get("path") or obj.get("filename") or obj.get("file")
                if path:
                    messages_ordered.extend(handle_update_file(path, obj.get("content", "")))
            elif action in ("create_project", "create project", "createproject"):
                folder = obj.get("folder") or obj.get("name") or obj.get("project")
                files = obj.get("files", [])
                if folder and files:
                    messages_ordered.extend(handle_create_project(folder, files))
            elif action in ("create_folder", "create folder", "createfolder"):
                folder = obj.get("folder") or obj.get("name")
                if folder:
                    messages_ordered.append(create_folder_action(folder))
            # Other actions (run_file, search, etc.) already in messages — append as-is
            # by falling back to original messages list below

    # Merge: anything in `messages` not covered by interleaved (run_file, debug, etc.)
    file_action_types = {"create_file","update_file","create_files","create_project","create_folder"}
    for m in messages:
        if m.get("type") not in file_action_types and m.get("type") != "response":
            messages_ordered.append(m)

    return messages_ordered


# ------------------------------------------------------------
# Deep Project Analysis Endpoint
# ------------------------------------------------------------
class DeepAnalyzeRequest(BaseModel):
    project_path: str
    project_name: str = ""
    max_depth: int = 5
    include_tests: bool = True


def analyze_project_structure(project_path: str, max_depth: int = 5, include_tests: bool = True) -> dict:
    structure = {
        "total_files": 0, "total_directories": 0, "file_tree": [],
        "main_entry_points": [], "config_files": [], "dependencies": [],
        "technologies": [], "code_metrics": {"total_lines": 0, "code_lines": 0,
        "comment_lines": 0, "blank_lines": 0, "average_file_size": 0, "complexity": "Low"}
    }
    entry_point_patterns = ['app.py', 'main.py', 'index.js', 'server.js', 'main.ts', 'index.ts', 'App.js', 'Main.java', 'main.go', 'main.rs', 'index.php', 'run.py', 'application.py', 'api.py', 'app.js', 'index.tsx', 'app.tsx']
    config_patterns = ['package.json', 'tsconfig.json', 'jsconfig.json', 'requirements.txt', 'setup.py', 'pyproject.toml', 'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'composer.json', 'Gemfile', '.env', '.env.example', 'docker-compose.yml', 'Dockerfile', '.gitignore', 'README.md']

    def build_tree(current_path: str, relative_path: str, depth: int) -> list:
        if depth > max_depth: return []
        nodes = []
        try:
            entries = list(Path(current_path).iterdir())
            for entry in entries:
                if entry.name.startswith('.') or entry.name in ['node_modules', '__pycache__', 'venv', '.git', 'dist', 'build', 'coverage']: continue
                if not include_tests and ('.test.' in entry.name or '.spec.' in entry.name or entry.name == '__tests__'): continue
                rel_path = str(Path(relative_path) / entry.name) if relative_path else entry.name
                if entry.is_dir():
                    structure["total_directories"] += 1
                    children = build_tree(str(entry), rel_path, depth + 1)
                    if children or entry.name not in ['node_modules', '__pycache__', 'venv']:
                        nodes.append({"name": entry.name, "path": rel_path, "type": "directory", "children": children})
                elif entry.is_file():
                    structure["total_files"] += 1
                    if entry.name.lower() in entry_point_patterns: structure["main_entry_points"].append(rel_path)
                    if entry.name.lower() in config_patterns: structure["config_files"].append(rel_path)
                    size = entry.stat().st_size
                    code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb', '.php', '.c', '.cpp', '.h']
                    ext = entry.suffix.lower()
                    if ext in code_extensions:
                        try:
                            with open(entry, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    structure["code_metrics"]["total_lines"] += 1
                                    stripped = line.strip()
                                    if not stripped: structure["code_metrics"]["blank_lines"] += 1
                                    elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('<!--'): structure["code_metrics"]["comment_lines"] += 1
                                    else: structure["code_metrics"]["code_lines"] += 1
                        except: pass
                    lang_map = {'.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.jsx': 'React', '.tsx': 'React TS', '.java': 'Java', '.go': 'Go', '.rs': 'Rust', '.rb': 'Ruby', '.php': 'PHP', '.c': 'C', '.cpp': 'C++', '.h': 'C Header', '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS', '.json': 'JSON', '.xml': 'XML', '.sql': 'SQL', '.sh': 'Shell', '.md': 'Markdown'}
                    nodes.append({"name": entry.name, "path": rel_path, "type": "file", "language": lang_map.get(ext, ext.upper().replace('.', '')), "size": size})
        except Exception as e: print(f"Error reading {current_path}: {e}")
        return sorted(nodes, key=lambda x: (x['type'] != 'directory', x['name']))

    structure["file_tree"] = build_tree(project_path, "", 0)
    if structure["total_files"] > 0: structure["code_metrics"]["average_file_size"] = structure["code_metrics"]["total_lines"] / structure["total_files"]
    avg = structure["code_metrics"]["average_file_size"]
    if avg > 1000: structure["code_metrics"]["complexity"] = "Very High"
    elif avg > 500: structure["code_metrics"]["complexity"] = "High"
    elif avg > 200: structure["code_metrics"]["complexity"] = "Medium"

    pkg_json = Path(project_path) / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, 'r') as f:
                pkg = json.load(f)
                for name, version in pkg.get('dependencies', {}).items(): structure["dependencies"].append({"name": name, "version": str(version), "type": "production"})
                for name, version in pkg.get('devDependencies', {}).items(): structure["dependencies"].append({"name": name, "version": str(version), "type": "development"})
                structure["technologies"].append({"name": "Node.js", "category": "Runtime", "usage": "Detected from package.json"})
        except: pass

    req_txt = Path(project_path) / "requirements.txt"
    if req_txt.exists():
        try:
            with open(req_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        match = re.match(r'^([a-zA-Z0-9_-]+)([=<>!~]+)?(.+)?', line)
                        if match: structure["dependencies"].append({"name": match.group(1), "version": match.group(3) if match.group(3) else "latest", "type": "production"})
            structure["technologies"].append({"name": "Python", "category": "Runtime", "usage": "Detected from requirements.txt"})
        except: pass

    if (Path(project_path) / "Cargo.toml").exists(): structure["technologies"].append({"name": "Rust", "category": "Language", "usage": "Detected from Cargo.toml"})
    if (Path(project_path) / "go.mod").exists(): structure["technologies"].append({"name": "Go", "category": "Language", "usage": "Detected from go.mod"})

    return structure


@app.post("/deep_analyze")
async def deep_analyze_endpoint(req: DeepAnalyzeRequest):
    try:
        structure = analyze_project_structure(req.project_path, req.max_depth, req.include_tests)
        main_files = structure.get("main_entry_points", [])[:5]
        config_files = structure.get("config_files", [])[:10]
        deps = structure.get("dependencies", [])[:15]
        techs = structure.get("technologies", [])

        prompt = f"""Analyze this project with gemini-3-pro-preview. Project: {req.project_name or 'Unknown'} at {req.project_path}. Structure: {structure['total_files']} files, {structure['total_directories']} dirs. Entry points: {', '.join(main_files) if main_files else 'None'}. Configs: {', '.join(config_files) if config_files else 'None'}. Deps: {', '.join([d['name'] for d in deps]) if deps else 'None'}. Tech: {', '.join([t['name'] for t in techs]) if techs else 'None'}. LOC: {structure['code_metrics']['code_lines']}, Complexity: {structure['code_metrics']['complexity']}. Return JSON with projectGoal (2-3 sentences), issues (severity, description, suggestion), enhancements (category, title, description, priority, effort)."""

        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"max_output_tokens": 32000, "temperature": 0.2})
        ai_response = response.text.strip()

        project_goal = "Unable to determine project goal"
        issues, enhancements = [], []

        try:
            json_match = re.search(r'\{[\s\S]*\}', ai_response)
            if json_match:
                ai_data = json.loads(json_match.group())
                project_goal = ai_data.get('projectGoal', project_goal)
                issues = ai_data.get('issues', [])
                enhancements = ai_data.get('enhancements', [])
        except: project_goal = ai_response[:500]

        return {"project_name": req.project_name or Path(req.project_path).name, "project_path": req.project_path, "code_structure": {"total_files": structure["total_files"], "total_directories": structure["total_directories"], "file_tree": structure["file_tree"], "main_entry_points": structure["main_entry_points"], "config_files": structure["config_files"]}, "project_goal": project_goal, "issues": issues, "enhancement_ideas": enhancements, "technologies": structure["technologies"], "dependencies": structure["dependencies"], "code_metrics": structure["code_metrics"]}
    except Exception as e:
        print(f"Error in /deep_analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "VibeCoding AI Backend is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        msgs = process_user_message(request)
        return ChatResponse(messages=msgs)
    except Exception as e:
        return ChatResponse(messages=[{"type": "error", "text": f"Server error: {str(e)}"}])

# Optional /debug endpoint for file fixing (if needed)
class DebugRequest(BaseModel):
    file_path: str
    content: str
    error: Optional[str] = None


@app.post("/debug")
async def debug_endpoint(req: DebugRequest):
    try:
        # Validate syntax
        syntax_error, _ = validate_python_code(req.content, req.file_path)

        # Build intelligent prompt
        prompt = f"""
You are an expert programmer. Fix the following code completely.
Return ONLY the corrected full file, no explanations, no markdown, no backticks.
Keep functionality intact, fix syntax/runtime errors, and address logical issues if obvious.

File: {req.file_path}
Terminal Error: {req.error if req.error else 'Not provided'}
Syntax Error: {syntax_error if syntax_error else 'None'}

Code:
{req.content}

Corrected code:
"""
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"max_output_tokens": 32000, "temperature": 0.2})
        fixed_content = response.text.strip()

        # Remove markdown code fences if present
        if fixed_content.startswith('```') and fixed_content.endswith('```'):
            lines = fixed_content.split('\n')
            # Remove first and last lines if they contain only backticks
            if lines and lines[0].strip().startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith('```'):
                lines = lines[:-1]
            fixed_content = '\n'.join(lines).strip()
        # Remove leading language identifier (e.g., "python\n")
        elif fixed_content.startswith('python\n'):
            fixed_content = fixed_content[7:]
        elif fixed_content.startswith('javascript\n'):
            fixed_content = fixed_content[11:]

        if not fixed_content:
            fixed_content = req.content  # fallback

        return {"fixed_content": fixed_content}
    except Exception as e:
        print(f"Error in /debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Image Analysis Endpoint
# ------------------------------------------------------------
class ImageAnalysisRequest(BaseModel):
    image: str  # base64 encoded image
    filename: str
    type: str = "screenshot"
    conversation_history: str = ""


@app.post("/analyze_image")
async def analyze_image_endpoint(req: ImageAnalysisRequest):
    try:
        # Build prompt for image analysis
        prompt = f"""
You are an expert code reviewer and debugger. Analyze this image (which appears to be a {req.type}) and provide insights.

If it's a screenshot of code:
- Identify any errors or bugs visible
- Suggest fixes or improvements
- Explain what the code does

If it's a UI screenshot:
- Describe the interface
- Suggest improvements or identify issues
- Provide relevant code suggestions if applicable

If it's an error message:
- Explain the error
- Provide the solution
- Give example code to fix it

Be concise but thorough in your analysis.
"""
        
        # For Gemini, we need to handle the image properly
        # The image is base64 encoded, we need to decode it
        import base64
        
        # Remove data URL prefix if present
        image_data = req.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Create image part for Gemini
        from google.genai import types
        
        # Determine mime type
        mime_type = "image/png"  # default
        if req.filename.lower().endswith('.jpg') or req.filename.lower().endswith('.jpeg'):
            mime_type = "image/jpeg"
        elif req.filename.lower().endswith('.png'):
            mime_type = "image/png"
        elif req.filename.lower().endswith('.gif'):
            mime_type = "image/gif"
        elif req.filename.lower().endswith('.webp'):
            mime_type = "image/webp"
        
        # Create content with image
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Use vision-capable model
            contents=[prompt, image_part]
        )
        
        analysis = response.text.strip()
        
        return {"analysis": analysis}
    except Exception as e:
        print(f"Error in /analyze_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# PDF Analysis Endpoint
# ------------------------------------------------------------
class PDFAnalysisRequest(BaseModel):
    pdf: str  # base64 encoded PDF
    filename: str
    conversation_history: str = ""


@app.post("/analyze_pdf")
async def analyze_pdf_endpoint(req: PDFAnalysisRequest):
    try:
        # For PDFs, we'll extract text and analyze it
        # Since we can't easily parse PDF in Python without additional libraries,
        # we'll send it to Gemini which can handle PDFs
        
        prompt = f"""
You are an expert document analyzer. Analyze this PDF document and provide a comprehensive summary.

Please:
1. Summarize the main content and purpose
2. Extract key information, requirements, or specifications
3. Identify any code examples, error messages, or technical details
4. Suggest how this document relates to the current project context

If this is a requirements document or specification:
- List the key requirements
- Suggest implementation approaches
- Identify potential challenges

Be thorough but concise in your analysis.
"""
        
        # Remove data URL prefix if present
        pdf_data = req.pdf
        if ',' in pdf_data:
            pdf_data = pdf_data.split(',')[1]
        
        # Decode base64
        import base64
        pdf_bytes = base64.b64decode(pdf_data)
        
        # Create PDF part for Gemini
        from google.genai import types
        
        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Use model that supports PDFs
            contents=[prompt, pdf_part]
        )
        
        analysis = response.text.strip()
        
        return {"analysis": analysis}
    except Exception as e:
        print(f"Error in /analyze_pdf: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ------------------------------------------------------------
# File Analysis Endpoint — used by ProjectAnalyzer
# Bypasses the chat pipeline entirely so JSON is never stripped
# ------------------------------------------------------------
class AnalyzeFileRequest(BaseModel):
    file_path: str
    content: str

@app.post("/analyze_file")
async def analyze_file_endpoint(req: AnalyzeFileRequest):
    """
    Dedicated endpoint for silent project analysis.
    Returns raw JSON with purpose + issues — never strips or modifies the response.
    """
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        # Aggressively truncate large files — we only need enough to understand purpose
        # 6000 chars ≈ ~150 lines, plenty for Gemini to understand what a file does
        content = req.content
        was_truncated = False
        if len(content) > 6000:
            content = content[:6000]
            was_truncated = True

        truncation_note = "\n[...file truncated for analysis...]" if was_truncated else ""

        prompt = f"""Analyze this file. Respond ONLY with a JSON object, nothing else.

File: {req.file_path}
```
{content}{truncation_note}
```

JSON format (no markdown, no explanation, raw JSON only):
{{"purpose": "one sentence describing what this file does", "issues": [{{"line": null, "description": "issue description", "severity": "warning"}}]}}

If no issues found, use empty array for issues. Output raw JSON only, starting with {{"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": 32000, "temperature": 0.2}
        )
        raw = response.text.strip()

        # Strip markdown fences if model disobeyed
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # Ensure it starts with { — strip any preamble
        brace_idx = raw.find('{')
        if brace_idx > 0:
            raw = raw[brace_idx:]

        # Parse and validate
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', raw)
                if match:
                    try:
                        parsed = json.loads(_sanitize_json_string(match.group()))
                    except:
                        raise HTTPException(status_code=422, detail=f"Could not parse AI response: {raw[:200]}")
                else:
                    raise HTTPException(status_code=422, detail=f"No JSON found in AI response: {raw[:200]}")

        return {
            "purpose": str(parsed.get("purpose", "")),
            "issues": parsed.get("issues", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Project Summary Endpoint — used by ProjectAnalyzer
# ------------------------------------------------------------
class AnalyzeSummaryRequest(BaseModel):
    file_list: str  # formatted list of "path: purpose" lines

@app.post("/analyze_summary")
async def analyze_summary_endpoint(req: AnalyzeSummaryRequest):
    """
    Generates overall project summary + enhancement suggestions.
    Bypasses chat pipeline — returns raw JSON directly.
    """
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        prompt = f"""You are a silent project analyzer. Based on these files, respond ONLY with valid JSON.

Files:
{req.file_list}

Respond with ONLY this JSON object (no markdown, no explanation, no code fences):
{{
  "summary": "2-3 sentence overview of what this project does.",
  "suggestedEnhancements": [
    "Enhancement suggestion 1",
    "Enhancement suggestion 2",
    "Enhancement suggestion 3"
  ]
}}"""

        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"max_output_tokens": 32000, "temperature": 0.2})
        raw = response.text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except:
                match = re.search(r'\{[\s\S]*\}', raw)
                parsed = json.loads(_sanitize_json_string(match.group())) if match else {}

        return {
            "summary": str(parsed.get("summary", "")),
            "suggestedEnhancements": parsed.get("suggestedEnhancements", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Project Query Endpoint — answers "find issues", "analyze project" etc.
# Uses .vibeproject.json data directly — no file scanning needed
# ------------------------------------------------------------
class ProjectQueryRequest(BaseModel):
    query: str
    project_data: dict   # full .vibeproject.json contents passed from extension

@app.post("/project_query")
async def project_query_endpoint(req: ProjectQueryRequest):
    """
    Answers natural language questions about the project using
    already-analyzed .vibeproject.json data.
    Never tries to list_files or search_folders.
    Returns a plain markdown response string.
    """
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        data = req.project_data
        project_name = data.get("projectName", "Unknown")
        summary      = data.get("summary", "")
        tech_stack   = ", ".join(data.get("techStack", []))
        files        = data.get("files", [])
        enhancements = data.get("suggestedEnhancements", [])

        # Build a rich context block from the JSON
        file_lines = []
        for f in files:
            issues = f.get("issues", [])
            issue_str = ""
            if issues:
                issue_str = " | Issues: " + "; ".join(
                    f"[{i.get('severity','?').upper()}] {i.get('description','')}"
                    for i in issues
                )
            deps = ", ".join(f.get("dependencies", [])[:5])
            file_lines.append(
                f"  - {f['path']} (health:{f.get('healthScore',100)}) — "
                f"{f.get('purpose','not analyzed yet')}"
                f"{(' | deps: ' + deps) if deps else ''}"
                f"{issue_str}"
            )

        context = f"""PROJECT: {project_name}
TECH STACK: {tech_stack}
SUMMARY: {summary}
OVERALL HEALTH: {data.get('totalHealthScore', 100)}/100
TOTAL FILES: {len(files)}

FILES ANALYZED:
{chr(10).join(file_lines)}

SUGGESTED ENHANCEMENTS:
{chr(10).join('  - ' + e for e in enhancements[:5]) if enhancements else '  (none yet)'}
"""

        prompt = f"""You are an expert code reviewer. A developer asked: "{req.query}"

Here is the complete analysis of their project from our analyzer:

{context}

Answer their question thoroughly using ONLY the data above.
- List specific issues by file and severity
- Suggest concrete fixes for each issue
- Mention dependencies that could be problematic
- Comment on health scores if relevant
- Be direct and actionable
- Use markdown formatting with headers and bullet points
- Do NOT say you need to scan files or use any tools — all data is already provided above
- Do NOT output any JSON action blocks"""

        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"max_output_tokens": 32000, "temperature": 0.2})
        answer = response.text.strip()

        # Strip any JSON action blocks the model accidentally emits
        answer = re.sub(r'\{[^{}]*?"action"[^{}]*?\}', '', answer, flags=re.DOTALL).strip()

        return {"response": answer}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /project_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ------------------------------------------------------------
class SmartEditRequest(BaseModel):
    prompt: str  # full surgical prompt built by editPlanner.ts

@app.post("/smart_edit")
async def smart_edit_endpoint(req: SmartEditRequest):
    """
    Receives a pre-built surgical edit prompt from the TypeScript EditPlanner.
    Returns a JSON object with patches, newImports, summary, and sideEffects.
    The extension applies the patches surgically — never rewrites the whole file.
    """
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        # Enforce that the model only returns JSON — no prose, no fences
        system_instruction = (
            "You are a surgical code patch generator. "
            "You MUST respond with ONLY a valid JSON object — no markdown, no explanation, no code fences. "
            "The JSON must have exactly these keys: "
            "\"summary\" (string), \"patches\" (array of {search, replace, description}), "
            "\"newImports\" (array of strings), \"sideEffects\" (string). "
            "The \"search\" value in each patch must be an EXACT verbatim copy from the provided file content. "
            "Never rewrite the whole file. Only patch what is necessary."
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\nConversation history:\n{conversation_history}\n\nUser: {user_input}\nAssistant:"
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config={"max_output_tokens": 32000, "temperature": 0.2}
        )
        return response.text.strip()

        # Strip markdown fences if AI disobeyed instructions
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # Validate it's parseable JSON before returning
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try sanitize then parse
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except json.JSONDecodeError:
                # Last resort: extract JSON object
                match = re.search(r'\{[\s\S]*\}', raw)
                if match:
                    parsed = json.loads(_sanitize_json_string(match.group()))
                else:
                    raise HTTPException(status_code=422, detail="AI returned non-JSON response")

        # Ensure required keys exist
        result = {
            "summary": parsed.get("summary", ""),
            "patches": parsed.get("patches", []),
            "newImports": parsed.get("newImports", []),
            "sideEffects": parsed.get("sideEffects", "")
        }

        # Validate patches structure
        clean_patches = []
        for p in result["patches"]:
            if isinstance(p, dict) and "search" in p and "replace" in p:
                clean_patches.append({
                    "search": str(p["search"]),
                    "replace": str(p["replace"]),
                    "description": str(p.get("description", ""))
                })
        result["patches"] = clean_patches

        return {"result": json.dumps(result)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /smart_edit: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ── Load GitHub credentials from .env (already loaded at top of backend.py) ──
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")


# ── Pydantic model for the token exchange request ─────────────────────────────
class GithubTokenRequest(BaseModel):
    code: str
    redirect_uri: str


# ── /github/token — exchanges OAuth code for access token ────────────────────
# Called by OAuthHandler.ts after user logs in via browser.
# Client Secret stays HERE in .env — never in the published extension. ✅
@app.post("/github/token")
async def github_token_exchange(req: GithubTokenRequest):
    """
    Receives the OAuth code from the VS Code extension.
    Exchanges it with GitHub using Client Secret (stored safely in .env).
    Returns the access token to the extension.
    """
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth not configured. Add GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET to .env"
        )

    try:
        response = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={
                "client_id":     GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code":          req.code,
                "redirect_uri":  req.redirect_uri
            },
            timeout=15  # Hard timeout — no hanging (blackbox fix ✅)
        )

        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub OAuth request failed: {response.status_code}"
            )

        data = response.json()

        # GitHub returns error in body (not HTTP status) — check for it ✅
        if "error" in data:
            raise HTTPException(
                status_code=400,
                detail=f"GitHub OAuth error: {data.get('error_description', data['error'])}"
            )

        if "access_token" not in data:
            raise HTTPException(
                status_code=502,
                detail="GitHub did not return an access token"
            )

        # Return only the token — nothing else needed by extension
        return {"access_token": data["access_token"]}

    except HTTPException:
        raise
    except requests.Timeout:
        # Never leave caller hanging with no response (blackbox fix ✅)
        raise HTTPException(
            status_code=504,
            detail="GitHub OAuth request timed out. Please try again."
        )
    except Exception as e:
        print(f"Error in /github/token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── /health — used by OAuthHandler to check backend is running ───────────────
# OAuthHandler.ts calls this BEFORE starting the login flow
# so it never starts a flow that will fail halfway (blackbox fix ✅)
@app.get("/health")
async def health_check():
    """Simple health check. OAuthHandler uses this before starting login."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
