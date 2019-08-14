def highlighter(color):
    def highlight(string):
        return '\033[{}m{}\033[0m'.format(color, string)
    return highlight


black = highlighter(30)
red = highlighter(31)
green = highlighter(32)
yellow = highlighter(33)
blue = highlighter(34)
magenta = highlighter(35)
cyan = highlighter(36)
white = highlighter(37)

bold = highlighter(1)

# For dim to work in tmux, it must be correctly configured.
# If it doesn't already work, one option is to place the following
# in ~/.tmux.conf:
# set -sa terminal-overrides ",*:dim=\\E[2m"
# Also note that a sufficiently new version of tmux is required.
dim = highlighter(2)
