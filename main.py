import sys

from jero.app import JeroApp


jero = JeroApp(sys.argv)

sys.exit(jero.run())