import time
import board
import adafruit_dotstar

DOTSTAR_DATA = board.D5
DOTSTAR_CLOCK = board.D6


dots = adafruit_dotstar.DotStar(DOTSTAR_CLOCK, DOTSTAR_DATA, 3, brightness=0.1)

#Green (255, 0, 0)
#Blue (0, 255, 0)
#Red (0, 0, 255)
dots[0] = (0, 0, 255)
dots[1] = (255, 0, 0)
dots[2] = (0, 255, 0)
dots.show()
#dots.deinit()
