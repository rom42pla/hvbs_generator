from os.path import join

from generator import Generator

DATA_PATH = join(".", "data")
ENDPOINTS_JSON_PATH = join(DATA_PATH, "endpoints.json")
MODEL_PATH = join(DATA_PATH, "model")

g = Generator(telegram_bot_token="5337038203:AAFfCbjG2gdbQTmAsjykxpWV41iehpLyA-k",
              model_path=MODEL_PATH)
g.start_telegram_bot()
