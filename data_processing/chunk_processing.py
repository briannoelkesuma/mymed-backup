import re
import json

# Function to split markdown content into sections
def split_markdown_sections(markdown_text):
    # Remove the first navigation link
    markdown_text = re.sub(r'^- \[.*?\]\(.*?\)\n\n', '', markdown_text, count=1)
    
    # Remove everything after "Tipsa & dela artikeln"
    markdown_text = re.split(r'## Tipsa & dela artikeln', markdown_text)[0]
                                                                         
    sections = []
    current_section = {"title": None, "content": ""}

    # Extract introduction before first heading
    intro_match = re.split(r'(?=^## )', markdown_text, maxsplit=1, flags=re.MULTILINE)
    if len(intro_match) > 1:
        intro_text = intro_match[0].strip()
        if intro_text:
            sections.append({"title": "Lågt blodtryck - Introduktion", "content": intro_text})
        markdown_text = intro_match[1]
    
    lines = markdown_text.split("\n")
    for line in lines:
        heading_match = re.match(r'^(#{2,4})\s(.+)', line)
        
        if heading_match:
            if current_section["title"]:
                sections.append(current_section)
                current_section = {"title": None, "content": ""}
            
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            current_section["title"] = heading_text
        
        current_section["content"] += line + "\n"
    
    if current_section["title"]:
        sections.append(current_section)
    
    return sections

# Function to chunk sections while maintaining semantic structure
def chunk_sections(sections, max_words=500):
    chunks = []
    
    for section in sections:
        words = section["content"].split()
        if len(words) <= max_words:
            chunks.append(section)
        else:
            split_content = re.split(r'(###\s.+)', section["content"])  # Split at lower headings
            sub_chunks = []
            current_chunk = ""
            
            for part in split_content:
                if re.match(r'###\s.+', part):
                    if current_chunk:
                        sub_chunks.append(current_chunk.strip())
                        current_chunk = ""
                current_chunk += part + "\n"
            
            if current_chunk:
                sub_chunks.append(current_chunk.strip())
            
            for sub in sub_chunks:
                chunks.append({"title": section["title"], "content": sub})
    
    return chunks

# Function to process the markdown and extract chunks
def process_markdown(data):
    markdown_text = data["markdown"]
    metadata_url = data["metadata"]["url"]
    metadata_title = data["metadata"]["title"]
    metadata_description = data["metadata"]["description"]
    
    sections = split_markdown_sections(markdown_text)
    chunks = chunk_sections(sections)
    
    structured_chunks = []
    for idx, chunk in enumerate(chunks):
        content = chunk["content"]
        if idx == 0:
            content =  metadata_description + "\n\n" + content

        structured_chunks.append({
            "id": idx,
            "title": chunk["title"],
            "content": content,
            "source_url": metadata_url,
            "metadata": {
                "page_title": metadata_title, 
                "page_description": metadata_description
            }
        })
       
        if idx == 0:
            structured_chunks[0].get("content").replace(metadata_description + "\n" + chunk["content"], "")
            # structured_chunks.append(metadata_description + "\n" + chunk["content"])
    
    return structured_chunks

# # Example usage with provided JSON
# with open('../scraping/scraped_data_allparams.json', 'r') as file:
#     data = json.load(file)

# processed_chunks = process_markdown(data)

# # Output JSON for use
# output_json = json.dumps(processed_chunks, indent=4, ensure_ascii=False)
# print(output_json)