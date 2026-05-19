bwrap \
  --share-net \
  --ro-bind / / \
  --bind "$(pwd)" "$(pwd)" \
  --dev-bind /dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --bind-try "$HOME/.claude" "$HOME/.claude" \
  --bind-try "$HOME/.claude.json" "$HOME/.claude.json" \
  --bind-try "$HOME/.config/claude" "$HOME/.config/claude" \
  --bind-try "$HOME/.local/state/claude" "$HOME/.local/state/claude" \
  --bind-try "$HOME/.cache/pixi" "$HOME/.cache/pixi" \
  --setenv HOME "$HOME" \
  --setenv PATH "$PATH" \
  --setenv TERM "$TERM" \
  --new-session \
  claude --dangerously-skip-permissions
