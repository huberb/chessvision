Extracts piece positions from screenshots of chessboards and outputs the board in ascii

Screenshots are taken from lichess.org.

Run `python download.py` to download training data from lichess.

Example Image:

![Example](/example/example.png?raw=true "Screenshot")


```
$ python main.py --predict ./example/example.png
output:
. . r . . r . k
. . . . . p p .
p . . . . b . .
. p . . . N . .
. . . N R . . .
. . P q . . . P
P P . . . P P .
. . . . R . K .
```
