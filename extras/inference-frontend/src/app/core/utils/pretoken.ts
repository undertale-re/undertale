/**
 * Helpers for Undertale's pretoken assembly format, shared by every completion
 * type (maskedlm, function naming, ...). The `[NEXT]` token marks instruction
 * boundaries and is required by the model's instruction/argument position
 * embeddings — it is not specific to any one task.
 */

/**
 * Joins multi-line assembly input into a single `[NEXT]`-separated pretoken
 * string, ready to submit to the inference API. Empty lines are dropped.
 */
export function joinInstructionLines(text: string): string {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join(' [NEXT] ');
}

/**
 * Reverses `joinInstructionLines()` for display, replacing `[NEXT]` separators
 * with newlines. Safe to call on text that has no `[NEXT]` tokens (no-op).
 */
export function formatForDisplay(text: string): string {
  return text.replace(/\s*\[NEXT\]\s*/g, '\n');
}
