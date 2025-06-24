import os
import csv
import time
import logging
import asyncio
from dotenv import load_dotenv
import pandas as pd
from telethon import TelegramClient, errors
from typing import List, Optional

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

def get_channels(xlsx_path: str, max_channels: int) -> List[str]:
    """
    Read channel usernames from an Excel file.
    Returns a list of channel usernames (without '@'), limited to max_channels.
    """
    df = pd.read_excel(xlsx_path, header=None)
    return [str(c).strip().lstrip('@') for c in df[0].dropna().unique()][:max_channels]

async def download_media_with_retry(client: TelegramClient, media, path: str, retries: int = RETRY_LIMIT) -> bool:
    """
    Download media with retry logic. Returns True if successful, False otherwise.
    """
    for attempt in range(retries):
        try:
            await client.download_media(media, path)
            return True
        except Exception as e:
            logging.warning(f"Failed to download media to {path} (attempt {attempt+1}): {e}")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return False

async def fetch_channel_messages(client: TelegramClient, channel_username: str, max_messages: int) -> List[dict]:
    """
    Fetch messages and metadata from a Telegram channel.
    Returns a list of message dicts.
    """
    messages = []
    try:
        entity = await client.get_entity(channel_username)
        count = 0
        async for message in client.iter_messages(entity):
            if count >= max_messages:
                break
            messages.append({
                'id': message.id,
                'text': message.message,
                'date': message.date,
                'media': message.media,
                'channel_title': entity.title if hasattr(entity, 'title') else channel_username,
                'channel_username': channel_username
            })
            count += 1
        logging.info(f"Fetched {count} messages from @{channel_username}")
    except Exception as e:
        logging.error(f"Error fetching messages from @{channel_username}: {e}")
    return messages

async def process_and_save_messages(messages: List[dict], writer, client: TelegramClient, media_dir: str):
    """
    Process messages: download media, write to CSV with metadata.
    """
    for msg in messages:
        media_path = None
        if msg['media'] and hasattr(msg['media'], 'photo'):
            filename = f"{msg['channel_username']}_{msg['id']}.jpg"
            media_path = os.path.join(media_dir, filename)
            success = await download_media_with_retry(client, msg['media'], media_path)
            if not success:
                media_path = None
        writer.writerow([
            msg['channel_title'],
            '@' + msg['channel_username'],
            msg['id'],
            msg['text'],
            msg['date'],
            media_path
        ])

async def scrape_all_channels(client: TelegramClient, channels: List[str], csv_path: str, media_dir: str):
    """
    Orchestrate scraping for all channels, handling retries and logging.
    """
    os.makedirs(media_dir, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
        for channel in channels:
            for attempt in range(RETRY_LIMIT):
                try:
                    print(f"Scraping @{channel} (attempt {attempt+1}) ...")
                    messages = await fetch_channel_messages(client, channel, MAX_MESSAGES)
                    await process_and_save_messages(messages, writer, client, media_dir)
                    print(f"Finished @{channel}")
                    break
                except (errors.FloodWaitError, errors.RPCError, Exception) as e:
                    logging.error(f"Error scraping @{channel} (attempt {attempt+1}): {e}")
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                logging.error(f"Failed to scrape @{channel} after {RETRY_LIMIT} attempts.")

async def main():
    """
    Main entry point: start client, get channels, and scrape all.
    """
    channels = get_channels(CHANNELS_XLSX, MAX_CHANNELS)
    await client.start()
    await scrape_all_channels(client, channels, CSV_PATH, MEDIA_DIR)

if __name__ == '__main__':
    # Initialize Telegram client
    client = TelegramClient('scraping_session', API_ID, API_HASH)
    with client:
        client.loop.run_until_complete(main()) 