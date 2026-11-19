#!/usr/bin/env sh
# Universal All-Agents Skill Installer for Snowball
# Supports: Antigravity (AGY), Claude Code, OpenAI Codex CLI, Cursor, Windsurf, Cline, OpenCode
#
# Usage:
#   ./install.sh <target> [--project]
#
# Targets:
#   antigravity | agy   Install for Google Antigravity
#   claude              Install for Claude Code
#   codex               Install for OpenAI Codex CLI
#   cursor              Install for Cursor
#   windsurf            Install for Windsurf
#   universal           Install to .agents/skills/snowball (Antigravity, OpenCode, Cline, Roo, Codex)
#   all                 Install to all project-level agent directories (.agents, .claude, .cursor)

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SKILL_SRC="${SCRIPT_DIR}/skills/snowball"
TARGET="${1:-universal}"
SCOPE="${2:-global}"

if [ ! -d "${SKILL_SRC}" ]; then
  echo "Error: Skill source directory not found at ${SKILL_SRC}" >&2
  exit 1
fi

install_to() {
  DEST="$1"
  mkdir -p "$(dirname "${DEST}")"
  rm -rf "${DEST}"
  cp -R "${SKILL_SRC}" "${DEST}"
  echo "Installed Snowball skill -> ${DEST}"
}

case "${TARGET}" in
  antigravity|agy)
    if [ "${SCOPE}" = "--project" ]; then
      install_to "./.agents/skills/snowball"
    else
      install_to "${HOME}/.gemini/config/skills/snowball"
    fi
    ;;
  claude)
    if [ "${SCOPE}" = "--project" ]; then
      install_to "./.claude/skills/snowball"
    else
      install_to "${HOME}/.claude/skills/snowball"
    fi
    ;;
  codex)
    if [ "${SCOPE}" = "--project" ]; then
      install_to "./.codex/skills/snowball"
    else
      install_to "${HOME}/.codex/skills/snowball"
    fi
    ;;
  cursor)
    if [ "${SCOPE}" = "--project" ]; then
      install_to "./.cursor/skills/snowball"
    else
      install_to "${HOME}/.cursor/skills/snowball"
    fi
    ;;
  windsurf)
    if [ "${SCOPE}" = "--project" ]; then
      install_to "./.windsurf/skills/snowball"
    else
      install_to "${HOME}/.codeium/windsurf/skills/snowball"
    fi
    ;;
  universal|opencode|cline)
    install_to "./.agents/skills/snowball"
    ;;
  all)
    install_to "./.agents/skills/snowball"
    install_to "./.claude/skills/snowball"
    install_to "./.cursor/skills/snowball"
    ;;
  *)
    echo "Usage: $0 <antigravity|claude|codex|cursor|windsurf|universal|all> [--project]" >&2
    exit 1
    ;;
esac
