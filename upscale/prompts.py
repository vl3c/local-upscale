"""The Qwen-Image instructions. Each was tuned on real family photos; see docs/workflow.md."""

RESTORE = ("Restore this old photo into a sharp, clean, well-focused photograph: remove the noise, blur and "
           "JPEG compression artifacts, and correct any color cast to natural, neutral white balance with "
           "realistic skin tones. Keep every person's face, identity, expression, hair and clothing, and the "
           "whole scene, exactly as they are.")

COLORIZE = ("Restore and colorize this old black-and-white photograph: repair scratches, dust, creases and "
            "fading, remove blur and noise, and make it sharp and clean. Add natural, realistic colors appropriate "
            "for the era, with realistic skin tones, hair and clothing colors, and natural daylight. Keep every "
            "person's face, identity, expression, hair and clothing, and the whole scene and composition, exactly "
            "as they are.")

# Used when --ref passes other photos of the same people. The references must only steer faces:
# without the explicit "never copy" clause Qwen sometimes returns a reference photo wholesale.
WITH_REFERENCES = ("The first image is the photo to restore: turn it into a sharp, clean, well-focused photograph, "
                   "remove the noise, blur and JPEG compression artifacts, and correct any color cast to natural, "
                   "neutral white balance with realistic skin tones. The other image(s) show the same people (maybe "
                   "at another time): use them only to get each person's real facial features right (face shape, "
                   "eyes, eyebrows, nose, mouth, ears, skin, hair). Never copy their background, clothes, pose, "
                   "framing or lighting. Keep the first image's composition, poses, expressions, clothing and the "
                   "whole scene exactly as they are.")

COLORIZE_WITH_REFERENCES = COLORIZE + (" The other image(s) show the same people: use them only to get their real "
                                       "facial features and colouring right, never copy anything else from them.")


def pick(colorize: bool, refs: bool) -> str:
    if colorize:
        return COLORIZE_WITH_REFERENCES if refs else COLORIZE
    return WITH_REFERENCES if refs else RESTORE


def with_also(prompt: str, also: str) -> str:
    """An extra edit (variants --also) put before the restore, which then applies to everything else."""
    also = also.strip().rstrip(".") + "."
    return f"{also} Apart from that change: {prompt}"
