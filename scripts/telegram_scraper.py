import os
import csv
import time
import logging
import asyncio
from dotenv import load_dotenv
import pandas as pd
from telethon import TelegramClient, errors
from typing import Optional

# ------------------- CONFIGURATION -------------------
load_dotenv('.env')
API_ID = os.getenv('TG_API_ID')
API_HASH = os.getenv('TG_API_HASH')
PHONE = os.getenv('phone')
CHANNELS_XLSX = 'data/raw/channels_to_crawl.xlsx'
CSV_PATH = 'data/raw/telegram_data_combined.csv'
MEDIA_DIR = 'data/raw/photos/'
MAX_MESSAGES = 1000
MAX_CHANNELS = 5
RETRY_LIMIT = 3
RETRY_BACKOFF = 5  # seconds
LOG_PATH = 'data/raw/scraper.log'

# ------------------- LOGGING SETUP -------------------
logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

# ------------------- UTILITY FUNCTIONS -------------------
def get_channels(xlsx_path: str, max_channels: int) -> list:
    df = pd.read_excel(xlsx_path, header=None)
    return [str(c).strip().lstrip('@') for c in df[0].dropna().unique()][:max_channels]

async def download_media_with_retry(client, media, path, retries=RETRY_LIMIT):
    for attempt in range(retries):
        try:
            await client.download_media(media, path)
            return True
        except Exception as e:
            logging.warning(f"Failed to download media to {path} (attempt {attempt+1}): {e}")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return False

async def scrape_channel(client, channel_username: str, writer, media_dir: str):
    """Scrape messages from a single Telegram channel with error handling and retry."""
    for attempt in range(RETRY_LIMIT):
        try:
            entity = await client.get_entity(channel_username)
            channel_title = entity.title if hasattr(entity, 'title') else channel_username
            count = 0
            async for message in client.iter_messages(entity):
                if count >= MAX_MESSAGES:
                    break
                media_path = None
                if message.media and hasattr(message.media, 'photo'):
                    filename = f"{channel_username}_{message.id}.jpg"
                    media_path = os.path.join(media_dir, filename)
                    success = await download_media_with_retry(client, message.media, media_path)
                    if not success:
                        media_path = None
                writer.writerow([
                    channel_title,
                    '@' + channel_username,
                    message.id,
                    message.message,
                    message.date,
                    media_path
                ])
                count += 1
            logging.info(f"Scraped {count} messages from @{channel_username}")
            return
        except (errors.FloodWaitError, errors.RPCError, Exception) as e:
            logging.error(f"Error scraping @{channel_username} (attempt {attempt+1}): {e}")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    logging.error(f"Failed to scrape @{channel_username} after {RETRY_LIMIT} attempts.")

async def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    channels = get_channels(CHANNELS_XLSX, MAX_CHANNELS)
    await client.start()
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
        for channel in channels:
            print(f"Scraping @{channel} ...")
            await scrape_channel(client, channel, writer, MEDIA_DIR)
            print(f"Finished @{channel}")

# ------------------- MAIN ENTRY -------------------
if __name__ == '__main__':
    client = TelegramClient('scraping_session', API_ID, API_HASH)
    with client:
        client.loop.run_until_complete(main()) 