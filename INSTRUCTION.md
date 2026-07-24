# INSTRUCTION.md - Complete Beginner's Setup Guide

Welcome! This guide assumes you have **never** used Python, VS Code, Git, FastAPI, or any
AI API before. Every step is spelled out - there is no prior knowledge assumed. Just
follow the steps in order, top to bottom, and don't skip ahead.

> 💡 **How to use this guide:** Each section is numbered to match a step in the setup
> process. Complete each one before moving to the next. If something doesn't work, jump
> to [Section 23: Troubleshooting](#23-troubleshooting) - most problems are covered there.

---

## Table of contents

| # | Section |
|---|---|
| 1 | [Installing Python](#1-installing-python) |
| 2 | [Installing Visual Studio Code](#2-installing-visual-studio-code) |
| 3 | [Installing Git](#3-installing-git) |
| 4 | [Required VS Code extensions](#4-required-vs-code-extensions) |
| 5 | [Opening the project](#5-opening-the-project) |
| 6 | [Creating a virtual environment](#6-creating-a-virtual-environment) |
| 7 | [Installing dependencies](#7-installing-dependencies) |
| 8 | [Creating the .env file](#8-creating-the-env-file) |
| 9 | [Configuring OpenAI API keys](#9-configuring-openai-api-keys) |
| 10 | [Configuring Anthropic API keys](#10-configuring-anthropic-api-keys) |
| 11 | [Configuring Gemini API keys](#11-configuring-gemini-api-keys) |
| 12 | [Running the application](#12-running-the-application) |
| 13 | [Uploading documents](#13-uploading-documents) |
| 14 | [Configuring Telegram Bot](#14-configuring-telegram-bot) |
| 15 | [Configuring Gmail API](#15-configuring-gmail-api) |
| 16 | [Configuring Calendar API](#16-configuring-calendar-api) |
| 17 | [Using RAG](#17-using-rag) |
| 18 | [Using MCP](#18-using-mcp) |
| 19 | [Using AI Agents](#19-using-ai-agents) |
| 20 | [Using Voice Features](#20-using-voice-features) |
| 21 | [Using Vision Features](#21-using-vision-features) |
| 22 | [Running tests](#22-running-tests) |
| 23 | [Troubleshooting](#23-troubleshooting) |
| 24 | [FAQ](#24-faq) |
| 25 | [Security Best Practices](#25-security-best-practices) |
| 26 | [Recommended Next Steps](#26-recommended-next-steps) |

---

## 1. Installing Python

Python is the programming language this whole application is written in. You need
**Python 3.12 or newer**.

### 1.1 Windows

1. Go to **https://www.python.org/downloads/** in your browser.
2. Click the big yellow **"Download Python 3.12.x"** button.
3. Run the downloaded installer (`python-3.12.x-amd64.exe`).
4. **This step is critical:** on the very first installer screen, check the box at the
   bottom that says **"Add python.exe to PATH"**, then click **"Install Now"**.

   ```
   [ SCREENSHOT PLACEHOLDER: Python installer first screen with
     "Add python.exe to PATH" checkbox highlighted ]
   ```

5. Wait for installation to finish, then click **Close**.
6. Verify it worked. Press `Win + R`, type `cmd`, press Enter to open Command Prompt, then run:
   ```bash
   python --version
   ```
   You should see something like:
   ```
   Python 3.12.4
   ```
   If you instead see `'python' is not recognized...`, restart your computer and try
   again - the PATH change needs a restart to take effect on some systems.

### 1.2 macOS

1. Go to **https://www.python.org/downloads/** in your browser.
2. Click **"Download Python 3.12.x"** (it will detect macOS automatically).
3. Open the downloaded `.pkg` file and follow the installer prompts (Continue -> Continue ->
   Agree -> Install).

   ```
   [ SCREENSHOT PLACEHOLDER: macOS Python installer welcome screen ]
   ```

4. Open **Terminal** (press `Cmd + Space`, type "Terminal", press Enter) and verify:
   ```bash
   python3 --version
   ```
   You should see:
   ```
   Python 3.12.4
   ```

> **Note:** On macOS, the command is `python3`, not `python`. Use `python3` and `pip3`
> throughout this guide unless a virtual environment is active (Section 6 explains why).

### 1.3 Linux (Ubuntu/Debian)

Most modern distributions already have Python 3.12. Verify, and install if missing:
```bash
python3 --version
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```

---

## 2. Installing Visual Studio Code

Visual Studio Code (VS Code) is a free code editor. This is where you'll open the
project, edit files, and run terminal commands.

1. Go to **https://code.visualstudio.com/**.
2. Click the big blue **Download** button (it auto-detects your operating system).
3. Run the installer:
   - **Windows:** run `VSCodeUserSetup-x64-x.xx.x.exe`. Accept the license, keep the
     default options, and importantly check **"Add to PATH"** if it's offered.
   - **macOS:** open the downloaded `.zip`, drag `Visual Studio Code.app` into your
     **Applications** folder.
   - **Linux:** install the downloaded `.deb` package: `sudo apt install ./code_*.deb`

   ```
   [ SCREENSHOT PLACEHOLDER: VS Code download page with the Download button highlighted ]
   ```

4. Open VS Code. You should see a welcome screen like this:

   ```
   [ SCREENSHOT PLACEHOLDER: VS Code welcome/start screen ]
   ```

---

## 3. Installing Git

Git is a tool for downloading and managing code from repositories (like GitHub). You need
it to clone this project (or you can skip Git and simply unzip the provided `.zip` file -
see the note below).

### 3.1 Windows
1. Go to **https://git-scm.com/download/win**. The download starts automatically.
2. Run the installer. Keep all default options and click **Next** through every screen,
   then **Install**.
3. Verify in Command Prompt:
   ```bash
   git --version
   ```
   Expected output: `git version 2.4x.x.windows.1`

### 3.2 macOS
1. Open Terminal and run:
   ```bash
   git --version
   ```
2. If Git isn't installed, macOS will prompt you to install the "Command Line Developer
   Tools" - click **Install** and wait for it to finish.

### 3.3 Linux
```bash
sudo apt install git -y
git --version
```

> **📦 Don't want to use Git?** If you received this project as a `.zip` file, you can
> skip Git entirely - just right-click the zip and **"Extract All"** (Windows) or
> double-click it (macOS) to unpack the folder, then continue from Section 5.

---

## 4. Required VS Code extensions

Extensions add features to VS Code. Open VS Code and click the **Extensions** icon in the
left sidebar (it looks like four squares, or press `Ctrl+Shift+X` / `Cmd+Shift+X`).

```
[ SCREENSHOT PLACEHOLDER: VS Code left sidebar with the Extensions icon highlighted ]
```

Search for and install each of these (click **Install** on each):

| Extension name | Publisher | Why you need it |
|---|---|---|
| Python | Microsoft | Syntax highlighting, autocomplete, and lets VS Code run/debug Python |
| Pylance | Microsoft | Fast, smart Python code analysis (usually installs automatically with Python) |
| Even Better TOML | tamasfe | Nicely formats `pytest.ini`/config files |
| SQLite Viewer | Florian Klampfer | Lets you browse the app's `data/app.db` database file visually |

```
[ SCREENSHOT PLACEHOLDER: Extensions search box showing "Python" extension by Microsoft
  with the Install button visible ]
```

---

## 5. Opening the project

### Option A - You have a `.zip` file
1. Extract the zip anywhere convenient, e.g. `Documents/ai-assistant-platform`.
2. In VS Code: **File -> Open Folder...** and select the extracted `ai-assistant-platform`
   folder.

### Option B - You're cloning from Git
1. Open VS Code.
2. Press `` Ctrl+` `` (or `` Cmd+` `` on Mac) to open the built-in terminal.
3. Navigate to where you want the project and clone it:
   ```bash
   cd Documents
   git clone <your-repository-url>
   cd ai-assistant-platform
   ```
4. **File -> Open Folder...** and select the `ai-assistant-platform` folder.

Once opened, you should see the file explorer on the left with files like `main.py`,
`config.py`, `requirements.txt`, and folders like `static/`, `templates/`, `tests/`.

```
[ SCREENSHOT PLACEHOLDER: VS Code Explorer sidebar showing the project's file list ]
```

---

## 6. Creating a virtual environment

A **virtual environment** is an isolated space for this project's Python packages, so
they don't clash with anything else on your computer. Always use one.

1. Open the terminal in VS Code: **Terminal -> New Terminal** (or `` Ctrl+` ``).
2. Make sure you're in the project folder - the prompt should show `ai-assistant-platform`.
3. Create the virtual environment:

   **Windows:**
   ```bash
   python -m venv .venv
   ```
   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   ```

   This creates a new hidden folder called `.venv` inside the project.

4. **Activate** the virtual environment:

   **Windows (Command Prompt):**
   ```bash
   .venv\Scripts\activate.bat
   ```
   **Windows (PowerShell):**
   ```bash
   .venv\Scripts\Activate.ps1
   ```
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

5. You'll know it worked when your terminal prompt starts with `(.venv)`, like this:
   ```
   (.venv) C:\Users\you\ai-assistant-platform>
   ```

   ```
   [ SCREENSHOT PLACEHOLDER: Terminal showing the (.venv) prefix after activation ]
   ```

> ⚠️ **PowerShell "running scripts is disabled" error?** Run this once, then try
> activating again:
> ```bash
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

> 💡 You must activate the virtual environment **every time** you open a new terminal to
> work on this project. VS Code often does this automatically if you select the right
> Python interpreter (see the tip below).

**Selecting the interpreter in VS Code:** Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac),
type "Python: Select Interpreter", and choose the one that mentions `.venv`.

```
[ SCREENSHOT PLACEHOLDER: Command Palette showing "Python: Select Interpreter" with
  the .venv option highlighted ]
```

---

## 7. Installing dependencies

With your virtual environment **activated** (you see `(.venv)` in the prompt), install
all required packages in one command:

```bash
pip install -r requirements.txt
```

This will take one to a few minutes the first time - it's downloading FastAPI, the
OpenAI/Anthropic/Gemini SDKs, LangChain, and everything else the app needs.

```
[ SCREENSHOT PLACEHOLDER: Terminal showing pip installing packages, ending with
  "Successfully installed ..." ]
```

**How to know it worked:** run this and confirm no errors appear:
```bash
python -c "import fastapi, openai, anthropic, google.generativeai; print('All core packages OK')"
```
Expected output:
```
All core packages OK
```

---

## 8. Creating the .env file

The `.env` file holds all your secret configuration (API keys, etc.) and is **never**
uploaded or shared. The project ships a template called `.env.example`.

1. Copy the template:

   **Windows:**
   ```bash
   copy .env.example .env
   ```
   **macOS/Linux:**
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in VS Code (click it in the file explorer, or run `code .env`).
3. You'll see a file that looks like this:

   ```env
   # --- Core app settings ---
   APP_NAME=AI Assistant Platform
   APP_ENV=development
   APP_SECRET_KEY=change-this-to-a-long-random-string
   APP_HOST=0.0.0.0
   APP_PORT=8000
   DATABASE_URL=sqlite+aiosqlite:///./data/app.db

   # --- Auth ---
   JWT_SECRET_KEY=change-this-too-please
   JWT_ALGORITHM=HS256
   JWT_EXPIRE_MINUTES=1440

   # --- AI Providers (set at least one) ---
   OPENAI_API_KEY=
   OPENAI_MODEL=gpt-4o-mini

   ANTHROPIC_API_KEY=
   ANTHROPIC_MODEL=claude-sonnet-4-6

   GOOGLE_API_KEY=
   GEMINI_MODEL=gemini-1.5-flash
   ```

4. Leave everything as-is for now - you'll fill in the API keys in the next three
   sections. Just make sure to replace `APP_SECRET_KEY` and `JWT_SECRET_KEY` with your
   own random strings (any long, random text works - mash your keyboard for 40 characters).

> ⚠️ **Never** commit or share your real `.env` file. It's already listed in
> `.gitignore` so Git won't accidentally track it.

---

## 9. Configuring OpenAI API keys

OpenAI provides the **GPT** models. This also powers voice (Whisper/TTS) and can power
vision.

1. Go to **https://platform.openai.com/signup** and create an account (or sign in).
2. Once logged in, go to **https://platform.openai.com/api-keys**.
3. Click **"+ Create new secret key"**.

   ```
   [ SCREENSHOT PLACEHOLDER: OpenAI API Keys page with "Create new secret key" button ]
   ```

4. Give it a name (e.g. "ai-assistant-platform"), click **Create secret key**.
5. **Copy the key immediately** - it starts with `sk-...` and OpenAI will only show it to
   you once.

   ```
   [ SCREENSHOT PLACEHOLDER: The generated key with a "Copy" button, and a warning
     that it won't be shown again ]
   ```

6. You will also need **billing set up** on your OpenAI account (Settings -> Billing) -
   the API is pay-per-use and does not work on a completely free account for most models.
7. Open your `.env` file in VS Code and paste the key:
   ```env
   OPENAI_API_KEY=sk-your-real-key-goes-here
   OPENAI_MODEL=gpt-4o-mini
   ```
8. Save the file (`Ctrl+S` / `Cmd+S`).

| Setting | What it does | Default |
|---|---|---|
| `OPENAI_API_KEY` | Your secret key | *(empty - required for OpenAI features)* |
| `OPENAI_MODEL` | Which GPT model powers chat | `gpt-4o-mini` |
| `OPENAI_TTS_MODEL` | Text-to-speech model | `tts-1` |
| `OPENAI_STT_MODEL` | Speech-to-text (Whisper) model | `whisper-1` |

---

## 10. Configuring Anthropic API keys

Anthropic provides the **Claude** models.

1. Go to **https://console.anthropic.com/** and sign up or sign in.
2. In the left sidebar, click **"API Keys"**.

   ```
   [ SCREENSHOT PLACEHOLDER: Anthropic Console sidebar with "API Keys" highlighted ]
   ```

3. Click **"Create Key"**, give it a name, and click **Create Key** again to confirm.
4. Copy the key (starts with `sk-ant-...`).
5. Add billing/credits under **Settings -> Billing** if prompted - like OpenAI, this is a
   paid API (Anthropic gives small free credits to new accounts in many regions).
6. Open `.env` and add:
   ```env
   ANTHROPIC_API_KEY=sk-ant-your-real-key-goes-here
   ANTHROPIC_MODEL=claude-sonnet-4-6
   ```
7. Save the file.

| Setting | What it does | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your secret key | *(empty - required for Claude features)* |
| `ANTHROPIC_MODEL` | Which Claude model powers chat | `claude-sonnet-4-6` |

---

## 11. Configuring Gemini API keys

Google provides the **Gemini** models, and the free tier here is usually the easiest way
to get started with zero cost.

1. Go to **https://aistudio.google.com/app/apikey**.
2. Sign in with a Google account.
3. Click **"Create API key"**.

   ```
   [ SCREENSHOT PLACEHOLDER: Google AI Studio "Create API key" button ]
   ```

4. Choose **"Create API key in new project"** if you don't already have a Google Cloud
   project.
5. Copy the generated key.
6. Open `.env` and add:
   ```env
   GOOGLE_API_KEY=your-real-gemini-key-goes-here
   GEMINI_MODEL=gemini-1.5-flash
   ```
7. Save the file.

| Setting | What it does | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Your secret key | *(empty - required for Gemini features)* |
| `GEMINI_MODEL` | Which Gemini model powers chat | `gemini-1.5-flash` |

> ✅ **You only need ONE of the three providers configured to start chatting.** Add more
> later whenever you like - the model switcher in the app only shows a provider as
> "configured" once its key is present.

---

## 12. Running the application

You're ready to launch the app!

1. Make sure your virtual environment is activated (`(.venv)` visible in the terminal -
   see Section 6 if not).
2. Run:
   ```bash
   uvicorn main:app --reload
   ```

   Or, for a fully guided start (checks Python, creates the venv, installs dependencies,
   checks `.env`, and launches automatically), just double-click:
   - **Windows:** `Start App.bat`
   - **macOS:** `Start App (Mac).command` (first time only, right-click -> Open, to bypass
     Gatekeeper's "unidentified developer" warning - or run `chmod +x "Start App (Mac).command"` once in Terminal)

3. You should see output like:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   INFO:     Application startup complete.
   ```

4. Open your web browser and go to:
   ```
   http://localhost:8000
   ```

   ```
   [ SCREENSHOT PLACEHOLDER: Browser showing the login page of the app ]
   ```

5. Click **"Create one"** to register your first account, fill in your name, email, and a
   password (minimum 8 characters), then sign in.

   ```
   [ SCREENSHOT PLACEHOLDER: The app's registration form ]
   ```

6. You're in! You should see the **Home** dashboard showing which AI providers are
   configured.

   ```
   [ SCREENSHOT PLACEHOLDER: Home dashboard showing OpenAI/Anthropic/Gemini status badges ]
   ```

7. To stop the server later, click the terminal and press `Ctrl+C`.

---

## 13. Uploading documents

This lets the assistant answer questions using your own PDF/TXT/Markdown files (this is
the RAG feature - more detail in [Section 17](#17-using-rag)).

1. In the left sidebar, click **Documents**.
2. Click **"Choose File"**, pick a `.pdf`, `.txt`, or `.md` file from your computer.
3. Click **"Upload & index"**.

   ```
   [ SCREENSHOT PLACEHOLDER: Documents page with a file selected and the
     "Upload & index" button ]
   ```

4. Wait a few seconds - the status badge will change from `processing` to `ready`. The
   file has now been split into chunks and embedded for search.

   ```
   [ SCREENSHOT PLACEHOLDER: Document library table showing a file with status "ready" ]
   ```

---

## 14. Configuring Telegram Bot

This lets you chat with your assistant from Telegram on your phone.

### 14.1 Create your bot
1. Open Telegram (app or web: https://web.telegram.org).
2. Search for the user **@BotFather** and start a chat with it.
3. Send the message: `/newbot`
4. Follow the prompts: give your bot a display name, then a unique username ending in
   `bot` (e.g. `my_ai_assistant_bot`).

   ```
   [ SCREENSHOT PLACEHOLDER: BotFather conversation showing the new bot's token ]
   ```

5. BotFather will reply with a token that looks like:
   ```
   123456789:AAExampleTokenTextGoesRightHere1234
   ```
   Copy this whole string.

### 14.2 Configure the app
1. Open `.env` and add:
   ```env
   TELEGRAM_BOT_TOKEN=123456789:AAExampleTokenTextGoesRightHere1234
   ```
2. Save the file.

### 14.3 Run the bot
The Telegram bot runs as its **own separate process**, alongside the web app (not part
of it). Open a **new** terminal (keep the web app running in the other one), activate the
virtual environment there too, then:
```bash
python telegram_service.py
```
You should see:
```
Starting Telegram bot polling loop...
```

4. Open Telegram, find your bot by the username you chose, and send it any message. It
   will reply using your platform's default AI model and remembers the conversation.

   ```
   [ SCREENSHOT PLACEHOLDER: A Telegram chat with the bot replying to "Hello!" ]
   ```

---

## 15. Configuring Gmail API

This lets the assistant read, summarize, draft, and send emails on your behalf.

### 15.1 Create a Google Cloud project and OAuth credentials
1. Go to **https://console.cloud.google.com/** and sign in.
2. Click the project dropdown at the top -> **"New Project"**. Name it (e.g.
   "ai-assistant-platform") -> **Create**.

   ```
   [ SCREENSHOT PLACEHOLDER: Google Cloud "New Project" dialog ]
   ```

3. With your new project selected, go to **APIs & Services -> Library**.
4. Search for **"Gmail API"**, click it, then click **Enable**.
5. Go to **APIs & Services -> OAuth consent screen**.
   - Choose **External** (unless you have a Google Workspace org), click **Create**.
   - Fill in an app name, your email as support email, and your email again as developer
     contact. Click **Save and Continue** through the remaining steps.
   - Under **Test users**, add your own Google account email so you can actually use it
     while the app is in "Testing" mode.

   ```
   [ SCREENSHOT PLACEHOLDER: OAuth consent screen configuration form ]
   ```

6. Go to **APIs & Services -> Credentials -> + Create Credentials -> OAuth client ID**.
   - Application type: **Web application**
   - Name: anything, e.g. "AI Assistant Platform"
   - Under **Authorized redirect URIs**, click **+ Add URI** and enter exactly:
     ```
     http://localhost:8000/api/gmail/oauth/callback
     ```
   - Click **Create**.

   ```
   [ SCREENSHOT PLACEHOLDER: OAuth client created dialog showing Client ID and
     Client Secret ]
   ```

7. Copy the **Client ID** and **Client Secret** shown.

### 15.2 Configure the app
Open `.env` and add:
```env
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/gmail/oauth/callback
```
Save the file, then restart the app (`Ctrl+C` then re-run `uvicorn main:app --reload`).

### 15.3 Connect your Gmail account
1. In the app, go to **Settings**.
2. Click **"Connect Gmail / Calendar"**.
3. A new browser tab opens with a Google sign-in and consent screen. Sign in and click
   **Allow** (you may see a "Google hasn't verified this app" warning since it's in
   Testing mode - click **Advanced -> Go to (your app name)** to proceed, since this is
   your own app).

   ```
   [ SCREENSHOT PLACEHOLDER: Google "unverified app" warning with Advanced link ]
   ```

4. Once approved, you're redirected back and Gmail is connected. Go to the **Emails**
   page to see your inbox summary and recent messages.

   ```
   [ SCREENSHOT PLACEHOLDER: Emails page showing a list of recent messages ]
   ```

---

## 16. Configuring Calendar API

Calendar uses the **same** Google Cloud project and OAuth credentials as Gmail - you
don't need to repeat Section 15's setup, just enable one more API.

1. Back in **Google Cloud Console -> APIs & Services -> Library**.
2. Search for **"Google Calendar API"**, click it, click **Enable**.

   ```
   [ SCREENSHOT PLACEHOLDER: Google Calendar API library page with Enable button ]
   ```

3. That's it - the same OAuth connection you approved in Section 15.3 already grants
   calendar access (the app requests both scopes together).
4. Go to the **Calendar** page in the app to view, create, and manage events, or try the
   **AI scheduling assistant**:
   ```
   e.g. type: "Set up a 30 min sync with Priya next Tuesday at 2pm about the roadmap"
   ```
   and click **"Parse into event"** to see it converted into a structured event you can
   create with one click.

   ```
   [ SCREENSHOT PLACEHOLDER: Calendar page showing the AI scheduling assistant result ]
   ```

---

## 17. Using RAG

**RAG** stands for **Retrieval-Augmented Generation** - instead of the AI answering only
from what it was trained on, it first *retrieves* relevant snippets from your own
documents, then *generates* an answer grounded in those snippets (with citations).

### How it works here (plain-English version)
| Step | What happens |
|---|---|
| 1. Upload | You upload a PDF/TXT/MD file on the **Documents** page |
| 2. Chunking | The file's text is split into ~800-character overlapping pieces |
| 3. Embedding | Each piece is converted into a list of numbers ("embedding") that represents its meaning |
| 4. Storage | Pieces + embeddings are saved in the local database |
| 5. Query | When you ask a question, your question is also embedded and compared against every stored piece using similarity math |
| 6. Answer | The most relevant pieces are inserted into the AI's context as `[Source N]`, and it answers using them |

### Try it
1. Upload a document (Section 13).
2. Go to **Chat**.
3. Click the **"Use documents (RAG)"** toggle chip above the message box so it turns
   active/highlighted.

   ```
   [ SCREENSHOT PLACEHOLDER: Chat page with the "Use documents (RAG)" toggle
     highlighted/active ]
   ```

4. Ask a question about the content of your uploaded document. The assistant's answer
   will be grounded in the document's actual text.
5. Alternatively, go to the **Documents** page and use the **"Ask your documents"** box
   directly to see the raw retrieved chunks and their similarity scores.

---

## 18. Using MCP

**MCP (Model Context Protocol)** is an open standard for letting AI applications talk to
external tools, data sources, and prompt templates in a consistent way - think of it as a
"USB-C port" for AI tools. This project includes both an MCP **server** (exposing tools)
and an MCP **client** (a demo of using them).

### 18.1 What's exposed
| Primitive | Name | What it does |
|---|---|---|
| Tool | `web_search` | Searches the web (requires `TAVILY_API_KEY`, optional) |
| Tool | `calculator` | Safely evaluates arithmetic expressions |
| Resource | `platform://about` | Static text describing the platform |
| Resource | `platform://changelog` | Points to the changelog |
| Prompt | `summarize_document` | Template for summarizing text |
| Prompt | `code_review` | Template for reviewing code |

### 18.2 Run the MCP server standalone
```bash
python mcp_server.py
```
This starts a server that communicates over **stdio** (standard input/output) using the
MCP protocol - it's meant to be launched by an MCP-compatible client (like Claude
Desktop), not opened directly in a browser.

### 18.3 Try the included demo client
In a terminal (with the virtual environment active):
```bash
python mcp_client.py
```
Expected output:
```
Available tools: ['web_search', 'calculator']
calculator(4+6)*2 -> {'result': 20}
Available resources: ['platform://about', 'platform://changelog']
Resource platform://about -> AI Assistant Platform: a multi-model, multi-agent assistant...
Available prompts: ['summarize_document', 'code_review']
Prompt result -> Summarize the following text into 5 concise bullet points: ...
```
This proves the client successfully launched the server as a subprocess and exercised all
three MCP primitives - tools, resources, and prompts.

> ⚠️ **Technical note:** MCP's stdio transport reserves **stdout** exclusively for
> protocol messages. If you ever modify `mcp_server.py`, never `print()` or log to stdout
> inside it - always use `stderr` (this is already configured correctly out of the box).

### 18.4 Connecting an external MCP client (e.g. Claude Desktop)
Add an entry like this to that application's MCP server configuration, pointing at your
project folder:
```json
{
  "mcpServers": {
    "ai-assistant-platform": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/full/path/to/ai-assistant-platform"
    }
  }
}
```

---

## 19. Using AI Agents

An **agent** is an AI that can plan multiple steps, call tools, observe the results, and
keep going until it reaches a final answer - instead of a single one-shot reply.

1. In the app, go to **Agents**.
2. Choose an agent type:

   | Agent | Best for |
   |---|---|
   | Research agent | Questions needing web search + synthesis across sources |
   | Coding agent | Planning and implementing a coding task |
   | Email agent | Drafting a message given a goal |

3. Type a goal, e.g.:
   ```
   Research the current landscape of open-source vector databases and summarize the top 3 options.
   ```
4. Click **"Run agent"**. It may take 10-30 seconds - the agent is reasoning through
   multiple steps.

   ```
   [ SCREENSHOT PLACEHOLDER: Agents page showing a final answer with the
     "Show reasoning steps" section expanded ]
   ```

5. Click **"Show reasoning steps"** to see exactly what the agent thought and which tools
   it called along the way - this transparency is one of the most useful parts of
   learning how agents work.

---

## 20. Using Voice Features

Voice features require an **OpenAI API key** (Section 9) - they use Whisper for
speech-to-text and OpenAI's TTS for text-to-speech.

1. Go to the **Voice** page.
2. **Speech to text:** click **"● Start recording"**, allow microphone access when your
   browser prompts you, speak, then click **"■ Stop recording"**. The transcript appears
   below.

   ```
   [ SCREENSHOT PLACEHOLDER: Browser microphone permission prompt ]
   [ SCREENSHOT PLACEHOLDER: Voice page showing a completed transcript ]
   ```

3. **Text to speech:** type any text into the box, choose a voice from the dropdown
   (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`), and click **"Generate speech"**.
   An audio player appears - click play to hear it.

---

## 21. Using Vision Features

Vision features work with either **OpenAI** (GPT-4o) or **Anthropic** (Claude) - whichever
you've configured; OpenAI is used by default if both are present.

1. Go to the **Vision** page.
2. Click **"Choose File"** and select an image (`.png`, `.jpg`, etc.).
3. Pick what you want to do:

   | Button | Result |
   |---|---|
   | Extract text (OCR) | Returns all text found in the image |
   | Describe image | Returns a one-sentence description |
   | Analyze receipt | Returns structured JSON: merchant, date, line items, total |

   ```
   [ SCREENSHOT PLACEHOLDER: Vision page with an uploaded receipt image and the
     structured JSON result shown below ]
   ```

---

## 22. Running tests

The project includes an automated test suite (pytest) that checks authentication,
conversation/memory/prompt CRUD operations, RAG math, and tool-calling safety - all
without needing any real API keys (it uses an in-memory test database).

1. Make sure your virtual environment is activated.
2. Run:
   ```bash
   pytest
   ```
3. Expected output (abbreviated):
   ```
   tests/test_api.py::test_register_and_login PASSED
   tests/test_api.py::test_login_rejects_wrong_password PASSED
   ...
   20 passed in 5.81s
   ```

   ```
   [ SCREENSHOT PLACEHOLDER: Terminal showing "20 passed" in green ]
   ```

4. For more detail on each test as it runs:
   ```bash
   pytest -v
   ```

---

## 23. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `'python' is not recognized as an internal or external command` | Python wasn't added to PATH | Reinstall Python and check "Add python.exe to PATH", or restart your computer |
| `ModuleNotFoundError: No module named 'fastapi'` (or similar) | Virtual environment not activated, or dependencies not installed | Activate `.venv` (Section 6), then `pip install -r requirements.txt` |
| Registration fails with a server error | A bcrypt/passlib version mismatch | Run `pip install bcrypt==4.0.1` inside your activated venv |
| Chat says "not configured" or errors when sending a message | No AI provider API key set | Add at least one key to `.env` (Sections 9-11) and restart the server |
| `Address already in use` / port 8000 busy | Another program (or a previous run) is using port 8000 | Stop the other process, or change `APP_PORT` in `.env` and run `uvicorn main:app --port 8001 --reload` |
| Gmail/Calendar pages show a 400 error | OAuth not connected yet | Go to **Settings -> Connect Gmail/Calendar** and complete the consent flow |
| "Google hasn't verified this app" warning during OAuth | Normal - your app is in Testing mode | Click **Advanced -> Go to (your app name)** since it's your own project |
| Telegram bot doesn't reply | Bot script isn't running, or wrong token | Run `python telegram_service.py` in its own terminal; double-check `TELEGRAM_BOT_TOKEN` in `.env` |
| MCP client hangs or throws a `BrokenResourceError` | Something is printing to stdout inside `mcp_server.py` | Only log to stderr in that file - don't add `print()` statements there |
| PowerShell won't let you activate the venv | Execution policy blocks scripts | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once |
| Voice/Vision features fail | Missing/invalid `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` for vision) | Double-check the key in `.env` and that billing is enabled on that provider account |
| Tests fail on a fresh clone | Dependencies not installed in the active environment | Re-run `pip install -r requirements.txt` inside the activated `.venv` |
| Changes to `.env` don't seem to apply | Server needs a restart to reload environment variables | Stop the server (`Ctrl+C`) and run `uvicorn main:app --reload` again |

Still stuck? Check the terminal output carefully - Python errors ("tracebacks") usually
name the exact file and line where something went wrong, and the last line of the error
is normally the most useful one to search for.

---

## 24. FAQ

**Q: Do I need all three AI providers (OpenAI, Anthropic, Gemini)?**
No - you only need one configured to start chatting. Add more any time.

**Q: Is this free to use?**
The app itself is free and open source. The AI providers (OpenAI, Anthropic, Google) bill
you per API call once you exceed any free tier/credits they offer. Gemini's free tier is
usually the most generous for experimentation.

**Q: Where is my data stored?**
Locally, in a SQLite database file at `data/app.db` inside the project folder. Nothing is
sent anywhere except to the AI provider APIs you've configured (and only when you send a
message, upload a doc for embedding, etc.).

**Q: Can I use this on my phone?**
The web app is responsive and works in a mobile browser if you're on the same network as
the computer running it (or you deploy it somewhere reachable). The Telegram bot
(Section 14) is the easiest fully-mobile experience.

**Q: What happens if I lose my `.env` file?**
You'll need to re-generate/re-enter your API keys and OAuth credentials - the `.env` file
itself is never uploaded anywhere, so there's no "cloud backup" to restore from by design.

**Q: Can multiple people use the same installation?**
Yes - each person registers their own account (Section 12, step 5) and gets their own
conversations, memory, documents, and preferences, all isolated from other accounts.

**Q: Why does the RAG feature still work without an OpenAI key?**
The app falls back to a simple local "hashing" embedding method when no OpenAI key is
present, so you can try the RAG feature end-to-end with zero API keys - just with lower
retrieval quality than real embeddings.

**Q: How do I stop the server?**
Click into the terminal running it and press `Ctrl+C`.

**Q: How do I update to a newer version of the code later?**
If you cloned with Git: `git pull`, then re-run `pip install -r requirements.txt` in case
dependencies changed. If you downloaded a zip, download the new zip and carefully copy
over your `.env` and `data/` folder so you don't lose your keys or database.

---

## 25. Security Best Practices

| Practice | Why it matters |
|---|---|
| Never commit or share your `.env` file | It contains your private API keys and secrets - anyone with them can spend your money or access your accounts |
| Replace `APP_SECRET_KEY` and `JWT_SECRET_KEY` with long random values | The defaults in `.env.example` are publicly known placeholder text, not real secrets |
| Don't reuse the same API key across unrelated projects | If one project is compromised, only that key needs to be revoked |
| Rotate/revoke API keys you're no longer using | Go to each provider's dashboard (OpenAI, Anthropic, Google) and delete unused keys |
| Only add Google OAuth test users you trust | While in "Testing" mode, anyone you add as a test user can authorize the app against their own Google account |
| Keep dependencies up to date | Run `pip list --outdated` periodically and update packages, especially security-sensitive ones like `bcrypt` and `pyjwt` |
| Don't expose the app directly to the internet without HTTPS | Put it behind a reverse proxy (nginx, Caddy) with TLS if you deploy it beyond your own machine |
| Review what each connected integration can access | Gmail/Calendar OAuth grants real read/send access to your account - only connect accounts you're comfortable with the app using |
| Back up `data/app.db` if it matters to you | It's a plain SQLite file - copy it somewhere safe periodically if your conversations/memory/documents are valuable |

---

## 26. Recommended Next Steps

Now that everything is running, here are good next things to explore:

1. **Read `ARCHITECTURE.md`** to understand how the pieces fit together under the hood.
2. **Read `API_REFERENCE.md`** to see every API endpoint the backend exposes - useful if
   you want to build your own frontend or script against it.
3. **Try the interactive API docs** at `http://localhost:8000/docs` - FastAPI
   auto-generates a full Swagger UI where you can test any endpoint directly in your
   browser.

   ```
   [ SCREENSHOT PLACEHOLDER: FastAPI's /docs Swagger UI page ]
   ```

4. **Experiment with the Prompt Library** (`/prompts`) - save your favorite prompts so
   you don't have to retype them.
5. **Set your default model and system prompt** on the **Settings** page so every new
   conversation starts exactly how you like.
6. **Try connecting an automation platform** (n8n, Zapier, or Make) by setting the
   relevant webhook URL in `.env` and calling `/api/automation/{platform}/trigger` - see
   `README.md` Section 10 for details.
7. **Read `CONTRIBUTING.md`** if you'd like to extend the project - add a new AI provider,
   a new tool, or a new integration.
8. **Read `SECURITY.md`'s hardening checklist** before using this anywhere beyond your own
   machine.
9. **Explore the code starting from `main.py`** - it's intentionally kept flat and
   readable, so tracing a request from `routes_chat.py` -> `chat_service.py` ->
   `llm_providers.py` is a great way to learn how a real FastAPI + multi-model AI app is
   put together.

You now have a fully working, multi-model AI assistant platform running on your own
machine. 🎉
