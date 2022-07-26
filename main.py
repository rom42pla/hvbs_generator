from os.path import join

from generator import Generator

DATA_PATH = join(".", "data")
ENDPOINTS_JSON_PATH = join(DATA_PATH, "endpoints.json")

g = Generator(endpoints_path=ENDPOINTS_JSON_PATH,
              telegram_bot_token="5337038203:AAFfCbjG2gdbQTmAsjykxpWV41iehpLyA-k")
g.start_telegram_bot()
g.start_apis()
