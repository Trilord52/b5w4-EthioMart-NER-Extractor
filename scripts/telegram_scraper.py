from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv
import pandas as pd
import asyncio

# Load environment variables
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# Read channel usernames from Excel (first column, strip @ if present)
channels_xlsx = 'data/raw/channels_to_crawl.xlsx'
df_channels = pd.read_excel(channels_xlsx, header=None)
channels = [str(c).strip().lstrip('@') for c in df_channels[0].dropna().unique()][:5]

# Output paths
csv_path = 'data/raw/telegram_data_combined.csv'
media_dir = 'data/raw/photos/'
os.makedirs(media_dir, exist_ok=True)

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title if hasattr(entity, 'title') else channel_username
        count = 0
        async for message in client.iter_messages(entity):
            if count >= 1000:
                break
            media_path = None
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                try:
                    await client.download_media(message.media, media_path)
                except Exception as e:
                    print(f"Failed to download media for {channel_username} {message.id}: {e}")
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
    except Exception as e:
        print(f"[Warning] Could not access channel @{channel_username}: {e}")

# Initialize the client
client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    # Open the CSV file and prepare the writer
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
        # Iterate over channels and scrape data into the single CSV file
        for channel in channels:
            print(f"Scraping @{channel} ...")
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from @{channel}")

if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main()) 