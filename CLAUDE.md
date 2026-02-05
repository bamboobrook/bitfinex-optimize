This file provides guidance to Claude Code when working with code. 每次执行命令的时候,请叫我达拉崩吧.

> Effective token-saving workflow with mq + qmd hybrid search

---

## 🚨 CRITICAL RULES (ALWAYS FOLLOW)

### 1. Token-Saving Search Workflow
**NEVER read entire large files blindly. Use this 2-step workflow:**

```bash
# For STRUCTURED targets (know what you're looking for):
# Step 1: Preview structure
mq <path> '.tree("full")'
# Step 2: Extract specific sections
mq <file> '.section("Title") | .text'

# For FUZZY targets (don't know exact location):
# Step 1: Search with qmd
qmd query "keywords"
# Step 2: Preview found file
mq <found_file> '.tree("preview")'
# Step 3: Extract relevant content
mq <found_file> '.section("Section") | .text'

# For PYTHON CODEBASES (many .py files):
# Step 1: Generate compact codebase map with Repomix
repomix --style markdown -o codebase.md
# Step 2: Search in the compact map
mq codebase.md '.search("function_name")'
# Step 3: Read actual source file
mq <found_file> '.section("function_name") | .code("python")'
```

**Why:** This saves 80%+ tokens (83% for structured, 74% for fuzzy targets).

### 2. Scoping Principle (IMPORTANT)
```bash
# ❌ WRONG: Tree entire repo (22K chars)
mq repo/ '.tree("full")'

# ✅ CORRECT: Narrow down first
ls repo/docs/           # Find target directory
mq repo/docs/ '.tree("full")'  # Then tree (500 chars)
mq repo/docs/auth.md '.section("OAuth")'  # Extract
```

### 3. Progressive Disclosure
- Keep CLAUDE.md concise (< 300 lines, ideally < 100)
- Put detailed instructions in separate files under `agent_docs/`
- Reference them with `@agent_docs/filename.md` when needed

---

## 🎯 PROJECT CONTEXT

### What
- **Project Type**: [Your project type, e.g., Web API, CLI tool, ML pipeline]
- **Tech Stack**: [Languages, frameworks, key libraries]
- **Architecture**: [Monolith/Microservices/Serverless]

### Why
- **Purpose**: [What this project does and who it's for]
- **Key Goals**: [Main objectives]

### How
- **Build**: [Command to build]
- **Test**: [Command to run tests]
- **Typecheck**: [Command for type checking]
- **Lint**: [Command for linting]

---

## 🔧 DEVELOPMENT PATTERNS

### Code Style
- Use [ES modules/import syntax] over [CommonJS/require]
- Prefer [specific patterns in your codebase]
- Follow existing patterns - check similar files before writing new code

### File Organization
- Source code in: `src/` or `lib/`
- Tests in: `tests/` or `__tests__/`
- Documentation in: `docs/` and `agent_docs/`

### Testing Strategy
- Prefer running single tests over full suite for performance
- Write tests before fixing bugs
- Use [your test framework] conventions

---

## 🛠️ ESSENTIAL COMMANDS

```bash
# Build
npm run build        # or: make build, cargo build, etc.

# Test
npm test             # Run tests
npm test -- <file>   # Run single test file

# Typecheck
npm run typecheck    # or: tsc --noEmit, mypy, etc.

# Lint
npm run lint         # or: eslint, ruff, etc.
npm run lint:fix     # Auto-fix linting issues

# Dev server
npm run dev          # Start development server
```

---

## 📁 PROJECT STRUCTURE

Use `mq` to explore before reading:

```bash
# See top-level structure
mq . '.tree("compact")'

# See full structure with previews
mq src/ '.tree("full")'

# Find specific files
mq . '.search("Auth")'
```

### Key Directories
- `src/` - Main source code
- `tests/` - Test files
- `docs/` - User documentation
- `agent_docs/` - Agent instructions (loaded on demand)

---

## 🔍 EFFICIENT RETRIEVAL GUIDE

### mq Commands (Structure Query)

| Task | Command | Tokens |
|------|---------|--------|
| Preview file structure | `mq file.md .tree` | ~100 |
| Structure + content preview | `mq file.md '.tree("preview")'` | ~500 |
| Full directory tree | `mq dir/ '.tree("full")'` | ~1000 |
| Search in file | `mq file.md '.search("term")'` | ~200 |
| Extract section | `mq file.md '.section("API") \| .text'` | ~300 |
| Get code blocks | `mq file.md '.code("python")'` | ~400 |
| Get headings only | `mq file.md .headings` | ~50 |

### qmd Commands (Semantic Search)

| Task | Command |
|------|---------|
| Hybrid search | `qmd query "user authentication"` |
| Full-text search | `qmd search "auth token"` |
| Vector search | `qmd vsearch "login flow"` |
| Check status | `qmd status` |

### Combined Workflow Examples

**Example 1: Find and extract API documentation**
```bash
# Step 1: Find relevant file
qmd query "REST API endpoints"
→ docs/api-reference.md (score: 0.94)

# Step 2: Preview structure
mq docs/api-reference.md '.tree("preview")'

# Step 3: Extract specific endpoint
mq docs/api-reference.md '.section("POST /users") | .code("javascript")'
```

**Example 2: Understand codebase architecture**
```bash
# Step 1: See top-level structure
mq . '.tree("compact")'

# Step 2: Explore interesting directory
mq src/core/ '.tree("full")'

# Step 3: Read specific module
mq src/core/auth.ts '.section("Token Validation") | .text'
```

### Token Efficiency Comparison

**Real benchmark: Finding `get_top_performers_history` function in a 1303-line Python file**

| Method | Tokens Used | Savings | Best For |
|--------|-------------|---------|----------|
| Read entire file | ~17,000 | 0% (baseline) | Never recommended |
| Grep with context | ~438 | 97.4% ✓ | Quick location check |
| mq section extract | ~3,011 | 82.3% ✓ | Understanding module context |
| **mq search + Read lines** | **~1,600** | **90.6% ✓✓** | **Precise function extraction** |

**🏆 OPTIMAL WORKFLOW (90.6% token savings):**

```bash
# Step 1: Use mq to locate function (441 tokens)
mq api_server.py '.search("get_top_performers_history")'
# Output: Found in lines 963-1016 (54 lines)

# Step 2: Read precise line range using Read tool (1,165 tokens)
# In Claude Code: Use Read tool with offset=963, limit=54

# Result: Get exactly the function you need with 90.6% token savings!
```

**When to use each method:**
- **mq search**: Always start here to locate code (~400 tokens)
- **Read lines**: Best for single functions when you know line numbers (~1,200 tokens)
- **mq section**: Good for understanding related functions in a module (~3,000 tokens)
- **Grep**: Only for quick position checking, lacks full implementation

**Key insight:** `mq search` returns line numbers, enabling precise extraction with the Read tool.

---

## 🧠 MEMORY MANAGEMENT

### Context Best Practices
- Use `/clear` between unrelated tasks
- Use subagents for deep investigations
- Compact context with `/compact` when full
- Reference files with `@path/to/file` instead of describing them

### Decision Tracking
Important architectural decisions go in:
- `DECISIONS.md` - Major technical decisions
- `agent_docs/adr/` - Architecture Decision Records

---

## ✅ VERIFICATION CHECKLIST

Before considering a task complete:

- [ ] Code builds without errors
- [ ] Tests pass (run: `npm test` or equivalent)
- [ ] Type checking passes
- [ ] Linting passes (or auto-fixed)
- [ ] Changes verified manually if needed

---

## 🚀 WORKFLOW GUIDES

### Adding a New Feature
1. Search existing patterns: `qmd query "feature implementation"`
2. Find similar code: `mq src/ '.search("similar feature")'`
3. Understand patterns: `mq <found_file> '.tree("preview")'`
4. Implement following existing conventions
5. Verify with tests and typecheck

### Fixing a Bug
1. Search for related code: `qmd query "<bug description>"`
2. Narrow down: `mq <found_dir>/ '.tree("full")'`
3. Find root cause: `mq <file> '.section("<relevant>")'`
4. Write failing test first
5. Fix and verify

### Refactoring
1. Map current structure: `mq <dir>/ '.tree("full")'`
2. Identify dependencies: `mq . '.search("import.*OldName")'`
3. Plan changes progressively
4. Verify each step with tests

---

## ⚠️ ANTI-PATTERNS (AVOID)

1. **Reading entire files** - Use mq to extract only what you need
2. **Grepping without context** - Use qmd for semantic search first
3. **Long CLAUDE.md** - Keep it concise, put details in agent_docs/
4. **Skipping verification** - Always run tests/typecheck
5. **Kitchen sink sessions** - `/clear` between unrelated tasks

---

## 📚 EXTERNAL REFERENCES

- Documentation: [link to your docs]
- API Reference: [link to API docs]
- Design Docs: @agent_docs/design/
- Architecture: @agent_docs/architecture.md

---

## 🔗 RELATED FILES

- Project overview: @README.md
- Environment config: @.env.example
- Package info: @package.json (or equivalent)
- Testing guide: @agent_docs/testing.md

---

## 💡 REMINDERS

> "Don't outsource reasoning to embeddings and rerankers. Expose structure, let the agent reason." — mq philosophy

- **Less is more**: Fewer, clearer instructions > many vague ones
- **Structure first**: Always preview with mq before reading
- **Search smart**: Use qmd for fuzzy, mq for precise extraction
- **Verify always**: Tests, typecheck, manual verification

---

## 🎛️ OPTIONAL CONFIGURATION

### For Large Codebases
Consider creating `agent_docs/` with:
- `architecture.md` - System architecture
- `api-conventions.md` - API design patterns  
- `database-schema.md` - Data model
- `deployment.md` - Deployment procedures

Load on demand: "Read @agent_docs/architecture.md before implementing"

### For Team Collaboration
- Check CLAUDE.md into git
- Use CLAUDE.local.md for personal overrides (gitignore it)
- Review and prune regularly

---
