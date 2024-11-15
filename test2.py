import locale
import sys

# Check system's preferred encoding
print("Preferred encoding:", locale.getpreferredencoding(False))

# Check Python's stdout encoding
print("Python stdout encoding:", sys.stdout.encoding)
