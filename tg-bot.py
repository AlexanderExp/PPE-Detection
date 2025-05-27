from mlpt.modules.ultralytics.ultralytics import YOLO
import torch
import numpy as np
import asyncio
import logging
import sys
import random
import string
import os
import cv2
from dotenv import load_dotenv
from os import getenv

from aiogram import Bot, Dispatcher, html, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, FSInputFile

# Bot token can be obtained via https://t.me/BotFather
load_dotenv()  # ищет .env в текущей папке
TOKEN = getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is not set")
bot = Bot(token=TOKEN)
model = YOLO("dvclive/artifacts/best.pt") #ТУТ ПУТЬ ДО BEST_PT

dp = Dispatcher()
router = Router()
dp.include_router(router)

checking_pairs = {
    "gloves": (9, 11),
    "ear-mufs": (2, 1),
    "face-guard": (4, 3),
    "face-mask": (5, 3),
    "glasses": (8, 3),
    "shoes": (14, 6),
    "helmet": (10, 12),
    "medical-suit": (13, 0),
    "safety-suit": (15, 0),
    "safety-vest": (16, 0)
}

defalut_config = list(checking_pairs.keys())


def check_image(config, image_path, path=None):  # config пришел от пользователя - на что проверять кадр
    ret = {}
    res = model(image_path)
    cls = res[0].boxes.cls.cpu().numpy()
    unique, counts = np.unique(cls, return_counts=True)
    clss = dict(zip(unique, counts))
    for entry in config:
        are, need = checking_pairs[entry]
        if need in clss.keys():
            if are in clss.keys():
                ret[entry] = clss[are] >= clss[need]
            else:
                ret[entry] = False
        else:
            ret[entry] = True
    if path is not None:
        res[0].save(filename=path)
    return ret  # dict, где на каждый элемент защиты True или False для этого кадра


@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer("Hi! Пришли мне фото или видео, а я подскажу, что не тут не так с точки зрения ТБ")


@router.message(lambda message: message.photo or message.video)
async def result_handler(message: Message) -> None:
    print("result_handler")
    if message.photo:
        await image_handler(message)
    elif message.video:
        await video_handler(message)
    else:
        await message.answer("Это не фото и не видео(")


async def image_handler(message: Message) -> None:
        image = message.photo[-1]
        filepath = "tmp/" + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".jpg"
        labeled_path = "tmp/" + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".jpg"
        open(filepath, "w")
        await bot.download(image.file_id, destination=filepath)
        check = check_image(defalut_config, filepath, labeled_path)
        answer = ""
        flag = False
        for elem, ok in check.items():
            if not ok:
                flag = True
                answer += f"Not enough: {elem}\n"
        if not flag:
            answer += "All good!"

        label = FSInputFile(labeled_path)
        await message.answer_photo(label)
        await message.answer(answer)
        os.remove(filepath)
        os.remove(labeled_path)



async def video_handler(message: Message) -> None:
    filepath = "tmp/" + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".mp4"
    with open(filepath, "w") as video_file:
        await bot.download(message.video.file_id, destination=filepath)
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = ""
        first_off = {item: -1 for item in defalut_config}
        while cap.isOpened():
            ex, frame = cap.read()
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(frame_num)
            if not ex:
                break
            check = check_image(defalut_config, frame)
            for elem, ok in check.items():
                if not ok and first_off[elem] == -1:
                    first_off[elem] = float(frame_num) / float(fps)
                if ok and first_off[elem] != -1:
                    result += f"No {elem} from {first_off[elem]}s to {frame_num / fps}s!\n"
                    first_off[elem] = -1
        if len(result) == 0:
            result = "All good!"
    rep_path = filepath.replace("mp4", "txt")
    with open(rep_path, "w") as text_file:
        text_file.write(result)
    logs = FSInputFile(rep_path)
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    await message.reply_document(logs)
    os.remove(rep_path)
    os.remove(filepath)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
