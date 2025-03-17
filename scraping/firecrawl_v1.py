import os
import time
import json
from requests.exceptions import HTTPError
from firecrawl import FirecrawlApp

# Firecrawl API Key
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

# Firecrawl and headers setup
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# List to store content for vector database
raw_documents = []

# Step 1: Scrape each link for detailed content
with open("Knowledge base/1177/v1.txt", "r") as file:
    links = file.readlines()

for link in links:
    link = link.strip()
    success = False

    while not success:
        try:
            scrape_result = firecrawl_app.scrape_url(link, params={'formats': ['markdown']})

            # Append the scrape result to raw_documents if successful
            raw_documents.append({
                "title": scrape_result['metadata']['title'], 
                "description": scrape_result['metadata']['description'], 
                "language": scrape_result['metadata']['language'],
                "markdown": scrape_result["markdown"]
            })
            success = True  # Exit loop after successful request

        except HTTPError as e:
            # Check if the error is due to rate limiting (429)
            if e.response.status_code == 429:
                print("Rate limit exceeded. Waiting for 15 seconds before retrying...")
                time.sleep(15)  # Wait 15 seconds before retrying
            else:
                print(f"Failed to scrape {link}: {e}")
                break  # Exit loop on other HTTP errors

print("Scraping completed!")

# Step 2: Save scraped content to JSON
with open("scraped_data_all.json", "w") as file:
    json.dump(raw_documents, file, indent=4)