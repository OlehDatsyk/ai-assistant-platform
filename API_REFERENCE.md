# API Reference

Interactive docs are also available at `/docs` (Swagger UI) and `/redoc` once the app is
running. This file is a quick-scan summary. All endpoints except `/health`, `/api/auth/register`,
and `/api/auth/login` require an `Authorization: Bearer <token>` header.

## Auth (`/api/auth`)
| Method | Path | Description |
|---|---|---|
| POST | `/register` | Create a user account |
| POST | `/login` | Exchange email+password for a JWT |
| GET | `/me` | Current user profile |

## Chat (`/api/chat`)
| Method | Path | Description |
|---|---|---|
| POST | `/conversations` | Create a conversation |
| GET | `/conversations` | List the user's conversations |
| GET | `/conversations/{id}/messages` | List messages in a conversation |
| PATCH | `/conversations/{id}/pin` | Toggle pinned state |
| DELETE | `/conversations/{id}` | Delete a conversation |
| POST | `/conversations/{id}/stream` | Send a message, stream the reply (SSE) |

## Memory (`/api/memory`)
| Method | Path | Description |
|---|---|---|
| GET | `` | List long-term memory items |
| POST | `` | Add a memory item |
| DELETE | `/{id}` | Delete a memory item |

## Prompt library (`/api/prompts`)
| Method | Path | Description |
|---|---|---|
| GET | `` | List saved prompts |
| POST | `` | Save a new prompt |
| PATCH | `/{id}/favorite` | Toggle favorite |
| DELETE | `/{id}` | Delete a prompt |

## Documents / RAG (`/api/documents`)
| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF/TXT/MD file (ingested in the background) |
| GET | `` | List uploaded documents |
| DELETE | `/{id}` | Delete a document and its chunks |
| POST | `/query` | Semantic search over one or more documents |

## Code assistant & agents (`/api`)
| Method | Path | Description |
|---|---|---|
| POST | `/code/run` | generate / explain / debug / refactor / review / test |
| POST | `/agents/run` | Run the research / coding / email agent |
| GET | `/models` | List providers and whether each is configured |

## Calendar (`/api/calendar`)
| Method | Path | Description |
|---|---|---|
| GET | `/events` | List upcoming events |
| POST | `/events` | Create an event |
| PATCH | `/events/{id}` | Update an event |
| DELETE | `/events/{id}` | Delete an event |
| GET | `/ai-schedule?text=...` | Parse natural language into a structured event |

## Gmail (`/api/gmail`)
| Method | Path | Description |
|---|---|---|
| GET | `/oauth/start` | Get the Google OAuth consent URL |
| GET | `/oauth/callback` | OAuth redirect target (handled automatically) |
| GET | `/emails` | List recent emails |
| GET | `/emails/search?q=...` | Search emails |
| GET | `/summary` | AI-generated inbox summary |
| POST | `/draft` | Draft a reply from intent |
| POST | `/send` | Send an email |

## Voice (`/api`)
| Method | Path | Description |
|---|---|---|
| POST | `/voice/transcribe` | Upload audio, get a transcript (Whisper) |
| POST | `/voice/speak?text=...` | Get synthesized speech (MP3 bytes) |

## Vision (`/api`)
| Method | Path | Description |
|---|---|---|
| POST | `/vision/ocr` | Extract text from an image |
| POST | `/vision/caption` | Describe an image |
| POST | `/vision/receipt` | Structured receipt extraction (JSON) |

## File search (`/api/files`)
| Method | Path | Description |
|---|---|---|
| GET | `/search?q=...&mode=content\|metadata` | Search the document library |

## Automation (`/api/automation`)
| Method | Path | Description |
|---|---|---|
| POST | `/{platform}/trigger?event=...` | Fire a webhook to n8n/Zapier/Make |

## Settings (`/api/settings`)
| Method | Path | Description |
|---|---|---|
| GET | `` | Get user preferences |
| PUT | `` | Update preferences (default model, theme, system prompt) |
