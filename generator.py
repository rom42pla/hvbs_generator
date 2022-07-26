import asyncio
import threading
from os.path import exists

import uvicorn
from fastapi import FastAPI
from aiogram import Bot, Dispatcher, executor, types

from data.db import Database
from apis.routers import items
from telegram_bot.commands import add_telegram_commands


class Generator:
    def __init__(self, endpoints_path: str,
                 telegram_bot_token: str):
        assert exists(endpoints_path)
        self.endpoints_path = endpoints_path
        self.db = Database(endpoints_path=self.endpoints_path)

        # APIs setup
        self.apis_app = FastAPI()
        for router, prefix in [
            (items.get_router(generator=self), "items"),
        ]:
            self.apis_app.include_router(
                router,
                prefix=f"/{prefix}",
                tags=[prefix],
            )

        def run_apis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            uvicorn.run(self.apis_app)

        self.apis_thread = threading.Thread(target=run_apis)

        # Telegram bot setup
        self.telegram_bot = Bot(token=telegram_bot_token)
        self.telegram_dispatcher = Dispatcher(self.telegram_bot)
        add_telegram_commands(self.telegram_dispatcher)

        def run_telegram_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            executor.start_polling(self.telegram_dispatcher, skip_updates=True)

        self.telegram_bot_thread = threading.Thread(target=run_telegram_bot)

    def start_apis(self):
        self.apis_thread.start()

    def start_telegram_bot(self):
        self.telegram_bot_thread.start()
