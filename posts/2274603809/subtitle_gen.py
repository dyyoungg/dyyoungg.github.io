# -*- coding: utf-8 -*-
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tqdm
from transformers import AutoTokenizer
import openai
from openai import OpenAI
import re
from openai import AsyncOpenAI
import asyncio
import time
from datetime import timedelta


class LLMconnector:
    def __init__(self, base_url) -> None:
        
        self.client = AsyncOpenAI(
            api_key='YOUR_API_KEY',
            base_url=base_url
        )

    async def generate(self, messages):

        model_cards = await self.client.models.list()._get_page()
        try:
            response = await self.client.chat.completions.create(
                model=model_cards.data[0].id,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=8192
            )
         
            return response.choices[0].message.content, response.usage.completion_tokens
        
        except openai.OpenAIError as e:
            return f"Error: {str(e)}"
        
def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    subtitles = re.findall(r'(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}\s*(.*?)\n', content, re.DOTALL)

    parsed_subtitles = []
    for subtitle in subtitles:
        start_time, end_time, text = subtitle
        
        text = ' '.join(text.splitlines()).strip()
        parsed_subtitles.append(f'{start_time}|{end_time}|{text}')

    return parsed_subtitles



async def fetch_answer(llm_teacher, prompt, text, parse_func):
    max_retry = 1
    count = 1
    while count <= max_retry:
        input_prompt = prompt.format(text)
        message = [{"role": "user", "content": input_prompt}]
        response, tokens = await llm_teacher.generate(message)
        results = parse_func(response)
        
        if results is not None:
            return results
        else:
            # print(text, response)
            count += 1
            print("retring...")
    return None




async def worker(queue, output_queue, llm_judge_base_url, save_dir):

    llm_judge = LLMconnector(llm_judge_base_url)
    tokenizer = AutoTokenizer.from_pretrained("/mnt/afs/share/Qwen25_72B_instruct")

    prompt = """你是一个游戏专家，你需要从游戏字幕文件中，提取与攻略相关的片段，并进行准确的总结。字幕中可能有游戏里面人物的声音，或者是视频制作者跟攻略无关的说话，请利用你的知识和逻辑进行判断。
    你需要首先判断这段字幕是否与攻略有关，如果无关，则返回"与游戏攻略无关"，如果有关，则需要总结润色这一段攻略内容，保证攻略的逻辑性，然后返回给我起始时间和你润色后的攻略内容，格式如下：
    ###
    起始时间|结束时间|你润色总结后的攻略内容
    ###
    以下是一些例子，帮助你更好的理解：
    ###
    例子1:
    开始时间|结束时间|字幕
    00:39:48|00:39:50|只能用紫色魂魄的一个技能
    00:39:50|00:39:53|具体可以看看这个演示画面
    00:39:53|00:39:55|广谋的话就是召唤毒蛇
    00:39:55|00:39:58|说实话这招光看能看出来非常拉胯
    00:39:58|00:40:00|幽魂是一个头锤
    00:40:00|00:40:02|我更偏向于前摇短的能力
    00:40:02|00:40:03|毕竟用的时候不是无敌的
    00:40:03|00:40:05|而且掉血也是会掉自己的
    00:40:05|00:40:07|每次更换之后会重新冷却
    00:40:07|00:40:09|记得一定要休息一下

    返回的内容:
    00:39:48|00:40:09|当前只能使用紫色魂魄的一个技能，推荐选择前摇短的技能，比如幽魂的头锤，因为它在战斗中更加灵活且适合快速反应，而广谋的召唤毒蛇技能表现较为一般；需要注意的是，使用技能时角色并非无敌状态，且会消耗自身血量，每次更换技能后都会重新冷却，因此使用后记得适当休息以调整状态。
    ###
    例子2:
    开始时间|结束时间|字幕
    00:44:09|00:44:10|而是好看的
    00:44:10|00:44:11|无伤很多时候无伤了
    00:44:11|00:44:12|但打的不满意的话
    00:44:12|00:44:14|照样要重录到满意为止
    00:44:14|00:44:16|无论是几十次还是几百次
    00:44:16|00:44:17|所以请耐心等待
    00:44:17|00:44:18|质量是有代价的
    00:44:18|00:44:20|所以大伙三连一下吧

    返回的内容:
    00:44:09|00:44:20|与游戏攻略无关
    ###
    以下是一个单机游戏黑神话悟空攻略视频的部分字幕，需要你按照我的要求和提供的例子工作
    {}

    返回的内容:
    """

    def parse_response(response):
        pattern = r'(\d{2}:\d{2}:\d{2})\|(\d{2}:\d{2}:\d{2})\|(.*)'
        matches = re.findall(pattern, response)
        results = []
        if matches:
            for match in matches:
                start_time, end_time, content = match
                if not "与游戏攻略无关" in content:
                    results.append({"start":start_time, "end": end_time, "content": content.strip()})
            return results
        else:
            return None

    def data_generator(subtitle_list, tokenizer, max_token=128):
        result = ""
        prefix = "开始时间|结束时间|字幕"
        result += prefix
        for line in subtitle_list:
            if len(tokenizer(result + line)["input_ids"]) > max_token:
                yield result + "\n" + line 
                result = prefix
            else:
                result += "\n" + line 

        if result:
            yield result

    while True:
        try:
            line = await queue.get()

            if line is None:
                queue.task_done()
                break
            
            subtitle_list = parse_srt(line)
            video_id = os.path.basename(line).split(".")[0]
            all_answers = []
            
            for text in data_generator(subtitle_list, tokenizer, max_token=128):

                task = asyncio.create_task(fetch_answer(llm_judge, prompt, text, parse_response))

                answer = await task

                if answer is not None:
                    all_answers += answer
            dump_data = {"video_id": video_id, "content": all_answers}
            save_path = os.path.join(save_dir, video_id+".json")
            await output_queue.put((dump_data, save_path))
            queue.task_done()

        except Exception as e:
            print(e)


async def display_progress(queue, total_files, start_time):
    while True:
        remaining_files = queue.qsize()
        processed_files = total_files - remaining_files
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        formatted_time = str(timedelta(seconds=int(elapsed_time)))  # Format the elapsed time
        print(f"\rProcessed {processed_files}/{total_files} files. Elapsed time: {formatted_time}", end='', flush=True)
        await asyncio.sleep(5)
        if remaining_files == 0:
            break

async def json_writer(output_queue):
    while True:
        item = await output_queue.get()
        if item is None:
            break
        data_dict, save_path = item
        if len(data_dict["content"])>0:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=4)
        output_queue.task_done()

async def RunworkerAsync(data_list, num_processes, save_dir, llm_base_url):

    os.makedirs(save_dir, exist_ok=True)
   
    queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    for file_path in data_list:
        queue.put_nowait(file_path)

    workers = []
    start_time = time.time()  # Record the start time
    progress_task = asyncio.create_task(display_progress(queue, total_files=len(data_list), start_time=start_time))
    writer_task = asyncio.create_task(json_writer(output_queue))

    for _ in range(num_processes):
        worker_task = asyncio.create_task(worker(queue, output_queue, llm_base_url, save_dir))
        workers.append(worker_task)

    await queue.join()

    for worker_task in workers:
        worker_task.cancel()
    
    progress_task.cancel()
    writer_task.cancel()

def get_data_list(root_dir, save_dir):
    srt_files = []
    for file in os.listdir(root_dir):
        if file.endswith('.json'):
            srt_files.append(file)
    
    existing_files = set([file for file in os.listdir(save_dir) if file.endswith('.json')])
  
    filtered_srt_files = [os.path.join(root_dir, file) for file in srt_files 
                          if file not in existing_files]
    
    filtered_srt_files.sort()
    
    return filtered_srt_files


def main(args):
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    llm_judge_base_url = f"http://0.0.0.0:{args.server_port}/v1"  
    num_processes = args.num_process
    
    data_root = args.data_root
    files = get_data_list(data_root, save_dir)

    chunk_index = args.chunk_index
    chunk_num = args.chunks_num
    chunk_start = max(0, int(chunk_index / chunk_num * len(files)))
    chunk_end = min(len(files), int((chunk_index + 1) / chunk_num * len(files)))
    data_list = files[chunk_start:chunk_end]

    asyncio.run(RunworkerAsync(data_list, num_processes, save_dir, llm_judge_base_url))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--data_root", type=str, default="/mnt/afs/yangdeyu/TTS/GameMLLM/LLaVA_hub/data/subtitle_qa/refined_subtitle")
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--chunks_num", type=int, default=1)
    parser.add_argument("--server_port", type=int, default=20033)
    parser.add_argument("--num_process", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="/mnt/afs/yangdeyu/TTS/GameMLLM/LLaVA_hub/data/subtitle_qa/subtitle_qa")
    args = parser.parse_args()
       
    main(args)

