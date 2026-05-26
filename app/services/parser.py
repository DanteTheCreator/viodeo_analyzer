"""COMMENT@ regex parser — used only by the /chat free-form endpoint.

AI-comment batch endpoints (/chat/post-ai-comments and /chat/oracle) use
JSON schema enforcement instead, making this parser unnecessary there.
"""

import re


def extract_auto_comments(text: str) -> list[tuple[float, str]]:
    """Parse COMMENT@<timestamp>: <text> lines from free-form model output.

    Handles:
      COMMENT@8.5: text          decimal seconds
      COMMENT@0:08: text         MM:SS colon format
      **COMMENT@8.5:** text      markdown bold wrapping
    Includes M.SS detection in case the model writes 0.17 to mean 17 seconds.
    """
    results = []
    for ts_raw, comment_text in re.findall(
        r"COMMENT@([\d:.]+)\s*:*\s*(.+)", text, re.IGNORECASE
    ):
        ts_raw = ts_raw.strip(":").strip()
        try:
            if ":" in ts_raw:
                parts = ts_raw.split(":")
                if len(parts) == 3:
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                else:
                    seconds = int(parts[0]) * 60 + float(parts[1])
                if seconds > 3600:
                    seconds = float(parts[-1])
            else:
                seconds = float(ts_raw)
                # M.SS detection: "0.17" means 17s, "0.22" means 22s
                dot_idx = ts_raw.find(".")
                if dot_idx >= 0 and seconds < 1.0:
                    dec_str = ts_raw[dot_idx + 1:]
                    if len(dec_str) >= 2:
                        sec_int = int(dec_str[:2])
                        if sec_int <= 59:
                            int_part = int(ts_raw[:dot_idx]) if ts_raw[:dot_idx].isdigit() else 0
                            sec_frac = float("0." + dec_str[2:]) if len(dec_str) > 2 else 0.0
                            seconds = int_part * 60 + sec_int + sec_frac
            results.append((seconds, comment_text.strip()))
        except (ValueError, IndexError):
            pass
    return results
