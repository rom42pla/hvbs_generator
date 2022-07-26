from aiogram import types


def add_telegram_commands(dp):
    @dp.message_handler(commands=['start', 'help'])
    async def send_welcome(message: types.Message):
        await message.reply("Succhiami il cazzo.")

    @dp.message_handler(commands=["anello"])
    async def send_welcome(message: types.Message):
        await message.answer("Anello di diamanti")
