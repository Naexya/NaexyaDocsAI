# Naexya Docs AI

Open-source AI-powered specification management tool that helps product and engineering teams collaborate with multiple large language models, extract structured requirements, and export professional documentation without leaving a browser tab.

---

## Features

- **Multi-provider AI integration** (OpenAI, Anthropic, Google, xAI, Moonshot, Qwen) with a unified client and per-provider rate limit awareness.
- **Dual AI personas** (Requirements Specialist + Technical Architect) designed to capture business context and technical design details in parallel.
- **Conversation-based specification extraction** that promotes iterative refinement and transparent traceability back to the originating chat history.
- **Validation workflow for quality control** so human reviewers can approve or reject generated specifications before they become canonical.
- **Professional export to HTML/Markdown** leveraging branded templates optimised for stakeholders and AI coding agents alike.
- **Local SQLite storage (no cloud dependencies)** providing self-hosted data retention with optional demo seed data for evaluation.
- **Bring-your-own-API-key model** ensuring you retain full control over model usage, quotas, and billing across all supported vendors.

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/NaexyaDocsAI.git
   cd NaexyaDocsAI
   ```
2. **Create an isolated environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Copy environment template and add your API keys**
   ```bash
   cp .env.example .env
   # Edit .env with your provider keys
   ```
5. **Launch the Gradio application**
   ```bash
   python app.py
   ```
6. **Open the local URL** printed by Gradio (typically `http://127.0.0.1:7860/`) to start collaborating with the personas and managing specifications.

> 💡 **Tip:** If you do not yet have API keys, enable the built-in demo data from the landing page. This allows you to explore the interface, validation queue, and export flows without making external API calls.

---

## Configuration

The platform is fully configurable through `config.py` and environment variables loaded from `.env`. Each provider entry defines endpoints, default parameters, and header templates to help you stay within rate limits.

### Environment Variables

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Secret key for OpenAI GPT-5 endpoints. |
| `ANTHROPIC_API_KEY` | Authentication token for Claude models. |
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini models. |
| `XAI_API_KEY` | API key for xAI Grok models. |
| `MOONSHOT_API_KEY` | Credential for Moonshot (Kimi) access. |
| `QWEN_API_KEY` | Access token for Alibaba Qwen models. |

> ⚠️ Keep the `.env` file out of version control. Only `.env.example` should be committed.

### Provider Setup Details

1. **OpenAI (GPT-5)**
   - Create or reuse an OpenAI account and generate a key from the dashboard.
   - Ensure the `https://api.openai.com/v1/chat/completions` endpoint is enabled for your organisation.
   - Optional parameters such as `temperature` and `max_tokens` can be fine-tuned in `config.py`.

2. **Anthropic (Claude-4-Sonnet)**
   - Request access to Claude-4 via the Anthropic console.
   - Place the key in `.env` as `ANTHROPIC_API_KEY`.
   - Respect the token per-minute limits published in the console; the defaults in `config.py` reflect conservative usage.

3. **Google (Gemini-2.5-Pro)**
   - Enable the Generative Language API in Google Cloud and create credentials through Google AI Studio.
   - Set `GOOGLE_API_KEY` and confirm the project has the `models.generateContent` permission.

4. **xAI (Grok-4-Fast)**
   - Obtain access from the xAI developer portal and generate an API key.
   - Update `.env` with `XAI_API_KEY`; the client automatically adds the `x-api-key` header required by Grok.

5. **Moonshot (Kimi-K2)**
   - Sign in to Moonshot AI, subscribe to the Kimi API plan, and generate a token.
   - Store the token in `MOONSHOT_API_KEY`; the client converts payloads to the Moonshot JSON schema for you.

6. **Qwen (Qwen3-Next)**
   - Activate DashScope and retrieve a key with text-generation permissions.
   - Save the key as `QWEN_API_KEY`; the integration handles the `Authorization: Bearer` header format.

After updating `.env`, restart the application so that Gradio reloads the configuration.

---

## Usage Guide

1. **Create or select a project** in the **Projects** tab. Each project stores conversations, specifications, and export history.
2. **Engage with the Requirements Specialist persona** in the **Requirements Chat** tab. Provide business objectives, user roles, and product scenarios. The assistant will log messages and surface candidate user stories.
3. **Switch to the Technical Architect persona** in the **Technical Chat** tab to capture APIs, data models, and system components with full technical depth.
4. **Review generated specifications** in the **Validation** tab. Approve high-quality outputs, request revisions, or reject items that need more context.
5. **Browse approved artefacts** in the **Specifications** tab. Filter by User Stories, Features, API Endpoints, Database Design, or System Architecture.
6. **Export documentation** from the **Export** tab. Download branded HTML or AI-friendly Markdown reports that include metadata, statistics, and links back to conversations.
7. **Manage provider settings** and rotate keys within the **Settings** tab. All changes are persisted locally so you can tailor the stack to your environment.

Throughout the workflow, the application captures timestamps and associations between conversations, personas, and specifications for full traceability.

---

## Deployment

### Local (Recommended for Development)

- Follow the Quick Start steps above.
- To run the app on a custom port, export `GRADIO_SERVER_PORT=XXXX` before launching `python app.py`.
- Use tools like `tmux` or `systemd` if you want to keep the application running in the background.

### Hugging Face Spaces

Hugging Face Spaces reads the [`config.yaml`](config.yaml) manifest and pinned
[`requirements.txt`](requirements.txt) to build and launch the application.

1. Create a new **Gradio** Space and connect it to your fork of the repository.
2. Review the metadata in `config.yaml`; update the title or colour palette if you fork the project.
3. Set the **Space hardware** to at least the default CPU (no GPU required).
4. In the Space **Variables** section, add any provider keys you plan to use
   (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`,
   `MOONSHOT_API_KEY`, `QWEN_API_KEY`). Spaces automatically exposes these as
   environment variables.
5. Optional: provide `NAEXYA_DEFAULT_PROVIDER` to specify which vendor should
   be called first when multiple keys are present.
6. Save the settings and rebuild the Space. Dependencies are installed from the
   pinned versions in `requirements.txt`, and `app.py` is used as the entry
   point.
7. Persistent storage is available under `/data`. The application automatically
   stores the SQLite database there when running inside a Space.

> 💡 No API keys yet? Launch the Space anyway. The interface automatically
> enters **demo mode** so you can explore the workflow using the built-in mock
> responses, validation queue, and exports without leaving the browser.

> 📘 For other hosting targets (e.g., Docker, Railway), reuse the same
> environment variables and ensure port `7860` is exposed.

---

## Contributing

We welcome pull requests and ideas from the community. To contribute:

1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-idea`).
2. Install dependencies and run the application locally to validate your changes.
3. Add or update documentation, including screenshots if you modify the UI.
4. Run `python -m compileall .` (or the relevant test suite once added) to ensure there are no syntax errors.
5. Submit a pull request describing the motivation, approach, and testing performed.

Please follow the existing coding style, docstring conventions, and commit message clarity when contributing.

---

## License

Naexya Docs AI is released under the terms of the [MIT License](LICENSE). You are free to self-host, extend, and integrate the project in accordance with the license.

