import re

CAMEL_TO_SNAKE_REGEX = re.compile(
    # Group to get the position before the enclosed match.
    # This position will be used for substituting underscores.
    r'('

    # Positive look behind
    # Only match if preceeded by a lowercase letter
    r'(?<=[a-z])'
    # Uppercase letter or digit
    r'[A-Z0-9]'

    # Alternatively, match the following pattern.
    # This captures situations where the string starts with a series of
    # uppercase letters. For instance: 'ROIPooling' -> 'roi_pooling'
    r'|'
    # # Not at the start of the line (Negative look ahead)
    r'(?!^)'
    # Uppercase letter or digit
    r'[A-Z0-9]'
    # Followed by lowercase letters (look ahead assertion)
    r'(?=[a-z])'

    r')'
)


def camel_to_snake(string):
    return CAMEL_TO_SNAKE_REGEX.sub(r'_\1', string).lower()


def indent(string, level):
    if level == 0:
        return string
    indentation = ' ' * (4 * level)
    return ''.join(
        indentation + line
        for line in string.splitlines(keepends=True)
    )
