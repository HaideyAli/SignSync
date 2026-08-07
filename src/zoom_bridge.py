"""Sends a finalized sentence toward the live meeting caption feed.

CLAUDE.md's original name (send_to_zoom_chat) implies Zoom's chat API, but
the actual mechanism for third-party live captions is different: the
meeting host enables "Use a third-party closed captioning service" in
Zoom's in-meeting settings, which generates a per-meeting URL of the form
https://wmcapi.zoom.us/rest/{path}?lang=en-US&seq={n} . POSTing plain-text
caption content to that URL, with `seq` incrementing on every call, injects
it as live captions -- not a chat message. The URL itself is the credential
(the host opts in per-meeting, no OAuth/app review needed) and must be
handled like a secret if the real call is ever wired in.

That HTTP call is NOT implemented here -- this stub exists so the rest of
the app (gui_app.py's "Send to Zoom" button) is wired end-to-end and ready
to receive the real POST once a meeting URL is available for testing.
"""
import logging

logger = logging.getLogger("zoom_bridge")


def send_to_zoom_chat(text: str) -> bool:
    """Stub matching the eventual real signature.
    TODO: POST `text` to the meeting's caption seq-URL (see module docstring).
    Returns True so callers can fire-and-forget without special-casing
    "not implemented" today.
    """
    text = text.strip()
    if not text:
        logger.warning("send_to_zoom_chat called with empty text — ignored")
        return False
    logger.info("Zoom bridge stub — would send caption: %r", text)
    print(f"[zoom_bridge stub] {text}")
    return True
