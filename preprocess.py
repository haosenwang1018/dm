import os
import json
from bs4 import BeautifulSoup
import re

# 新增：导入过滤函数
from data_utils import filter_documents

def extract_text_and_title_from_html(html_filepath):
    """
    从指定的 HTML 文件中提取标题和正文文本。
    Args:
        html_filepath (str): HTML 文件的路径。
    Returns:
        tuple: (标题, 正文文本) 或 (None, None) 如果失败。
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')

        # --- 提取标题 ---
        title_tag = soup.find('title')
        title_string = title_tag.string if title_tag else None
        title = title_string.strip() if title_string else os.path.basename(html_filepath)
        title = title.replace('.html', '')

        # --- 定位正文内容 ---
        content_tag = soup.find('content')
        if not content_tag:
            content_tag = soup.find('div', class_='rich_media_content')
        if not content_tag:
            content_tag = soup.find('article')
        if not content_tag:
            content_tag = soup.find('main')
        if not content_tag:
            content_tag = soup.find('body')

        if content_tag:
            text = content_tag.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n', text).strip()
            text = text.replace('阅读原文', '').strip()
            return title, text
        else:
            print(f"警告：在文件 {html_filepath} 中未找到正文标签。")
            return title, None

    except FileNotFoundError:
        print(f"错误：文件 {html_filepath} 未找到。")
        return None, None
    except Exception as e:
        print(f"处理文件 {html_filepath} 时出错: {e}")
        return None, None

def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    将文本分割成指定大小的块，并带有重叠。
    Args:
        text (str): 要分割的文本。
        chunk_size (int): 每个块的目标字符数。
        chunk_overlap (int): 相邻块之间的重叠字符数。
    Returns:
        list[str]: 文本块列表。
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text):
            break
        if start < chunk_size and len(chunks) > 1 and \
           chunks[-1] == chunks[-2][chunk_size-chunk_overlap:]:
            chunks.pop()
            start = len(text)

    if start < len(text) and start > 0:
        last_chunk = text[start-chunk_size+chunk_overlap:]
        if chunks and last_chunk != chunks[-1]:
            if not chunks[-1].endswith(last_chunk):
                chunks.append(last_chunk)
        elif not chunks:
            chunks.append(last_chunk)

    return [c.strip() for c in chunks if c.strip()]

# --- 配置 ---
html_directory    = './data/'
output_json_path  = './data/processed_data.json'
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 50

# --- 主处理逻辑 ---
all_data_for_milvus = []
file_count = 0
chunk_count = 0

print(f"开始处理目录 '{html_directory}' 中的 HTML 文件...")

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
html_files = [f for f in os.listdir(html_directory) if f.endswith('.html')]
print(f"找到 {len(html_files)} 个 HTML 文件。")

for filename in html_files:
    filepath = os.path.join(html_directory, filename)
    print(f"  处理文件: {filename} ...")
    file_count += 1

    title, main_text = extract_text_and_title_from_html(filepath)
    if main_text:
        chunks = split_text(main_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"    分割成 {len(chunks)} 个块。")
        for i, chunk in enumerate(chunks):
            chunk_count += 1
            milvus_entry = {
                "id": f"{filename}_{i}",
                "title": title or filename,
                "abstract": chunk,
                "source_file": filename,
                "chunk_index": i
            }
            all_data_for_milvus.append(milvus_entry)
    else:
        print(f"    警告：未能提取正文，跳过。")

print(f"\n处理完成。共 {file_count} 个文件，生成 {chunk_count} 个文本块。")

# ——— 在保存前进行文档过滤 ———
all_data_for_milvus = filter_documents(all_data_for_milvus, min_length=200)

# --- 保存为 JSON ---
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_for_milvus, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_json_path}")
except Exception as e:
    print(f"错误：无法写入 JSON 文件 {output_json_path}: {e}")
