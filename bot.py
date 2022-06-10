import uuid
import telebot
import os
import requests
import json


def main():
    token = os.environ["KUKURUZER_BOT_TOKEN"]
    bot = telebot.TeleBot(token)


    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        bot.reply_to(message, "Howdy, how are you doing?")


    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        bot.reply_to(message, message.text)


    @bot.message_handler(content_types=["photo"])
    def save_photos(message):
        filenames = []
        for ph in message.photo:
            path = json.loads(requests.get(f'https://api.telegram.org/bot{token}/getFile?file_id={ph.file_id}').text)["result"]["file_path"]
            filetype = path.split('.')[-1]
            response = requests.get(f'https://api.telegram.org/file/bot{token}/{path}')
            filename = f'{uuid.uuid1()}.{filetype}'
            open(filename, "wb").write(response.content)
            filenames.append(filename)


    # def send_photo(path):
    #     photo = open(path, 'rb')
    #     bot.send_photo(chat_id, photo)


    bot.infinity_polling()