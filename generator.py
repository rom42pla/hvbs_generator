import asyncio
import json
import threading
from os.path import exists, isdir

from aiogram import Bot, Dispatcher, executor, types
from transformers import pipeline


class Generator:
    def __init__(self,
                 telegram_bot_token: str,
                 model_path: str):
        assert isinstance(model_path, str) and isdir(model_path)
        self.model_path = model_path

        # model setup
        self.generator = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
        )

        # Telegram bot setup
        self.telegram_bot = Bot(token=telegram_bot_token)
        self.telegram_dispatcher = Dispatcher(self.telegram_bot)

        @self.telegram_dispatcher.message_handler(commands=['start', 'help'])
        async def send_welcome(message: types.Message):
            await message.reply("Succhiami il cazzo.")

        @self.telegram_dispatcher.message_handler(commands=["genera"])
        async def send_welcome(message: types.Message):
            message.text = message.text.replace("/genera", "", 1)
            numbers = {
                0: "0️⃣",
                1: "1️⃣",
                2: "2️⃣",
                3: "3️⃣",
                4: "4️⃣",
                5: "5️⃣",
                6: "6️⃣",
                7: "7️⃣",
                8: "8️⃣",
                9: "9️⃣",
            }
            out = ""
            for i_sentence, sentence in enumerate(self.generate_sentences(input_string=message.text, amount=4)):
                out += f"{numbers[i_sentence + 1]}\t{sentence}\n\n"
            await message.answer(out, disable_web_page_preview=True)

        def run_telegram_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            executor.start_polling(self.telegram_dispatcher, skip_updates=True)

        self.telegram_bot_thread = threading.Thread(target=run_telegram_bot)

    def start_telegram_bot(self):
        self.telegram_bot_thread.start()

    def generate_sentences(self, input_string: str = "", amount: int = 1):
        assert isinstance(amount, int) and amount >= 1
        sentences = []
        for sentence in [s["generated_text"].strip()
                         for s in self.generator(input_string,
                                                 do_sample=True,
                                                 top_k=50,
                                                 top_p=0.95,
                                                 max_length=64,
                                                 early_stopping=True,
                                                 num_return_sequences=amount)]:
            sentence = sentence.replace("\\n", "")
            sentence = sentence.replace("\\t", "")
            sentence = sentence.replace("  ", "")
            sentences += [sentence]
        return sentences
