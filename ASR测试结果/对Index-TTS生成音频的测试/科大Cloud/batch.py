# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import os
import time
import requests
import urllib.parse
import wave # For getting audio duration
import glob # For finding files

# --- API Configuration ---
lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'

class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path, audio_duration_seconds):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.audio_duration_seconds = audio_duration_seconds # Store actual audio duration
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa

    def upload(self):
        print(f"  上传文件: {os.path.basename(self.upload_file_path)}")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict["fileSize"] = file_len
        param_dict["fileName"] = file_name
        # Use actual duration rounded to nearest second as a string
        param_dict["duration"] = str(int(round(self.audio_duration_seconds)))
        # print("  upload参数:", param_dict) # Optional: for debugging

        try:
            with open(upload_file_path, 'rb') as f:
                data = f.read()
        except IOError as e:
            print(f"  错误: 读取文件失败 {upload_file_path}: {e}")
            return None

        try:
            response = requests.post(url=lfasr_host + api_upload + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"}, data=data, timeout=30) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"  错误: 上传请求失败: {e}")
            return None
        except json.JSONDecodeError:
            print(f"  错误: 解析上传响应JSON失败: {response.text}")
            return None
        
        # print("  upload_url:", response.request.url) # Optional: for debugging
        # print("  upload resp:", result) # Optional: for debugging
        return result

    def get_result(self):
        """
        Uploads the file and polls for the transcription result.
        Returns the concatenated transcribed text, or an empty string if an error occurs or no text is found.
        """
        uploadresp = self.upload()
        if not uploadresp or uploadresp.get("code") != "000000" or 'content' not in uploadresp or 'orderId' not in uploadresp['content']:
            print(f"  文件上传失败或返回结果格式不正确。")
            if uploadresp and 'descInfo' in uploadresp:
                print(f"  错误信息: {uploadresp.get('descInfo', '未知错误')}")
            elif uploadresp:
                 print(f"  上传响应: {uploadresp}")
            return "" # Return empty string on failure

        orderId = uploadresp['content']['orderId']
        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa # Re-using signa from upload, ensure API allows this for getResult
        param_dict['ts'] = self.ts       # Re-using ts from upload
        param_dict['orderId'] = orderId
        param_dict['resultType'] = "transfer,predict"
        
        print(f"  查询转写结果 (OrderID: {orderId})...")
        # print("  get result参数:", param_dict) # Optional: for debugging
        
        status = 3 # Initial status "processing"
        max_retries = 24 # Max polling attempts (e.g., 24 * 5s = 120s = 2 minutes)
        retries = 0

        final_result_json = None

        while status == 3 and retries < max_retries:
            try:
                response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                         headers={"Content-type": "application/json"}, timeout=20) # Added timeout
                response.raise_for_status()
                result_poll = response.json()
            except requests.exceptions.RequestException as e:
                print(f"  错误: 查询结果请求失败: {e}")
                time.sleep(5)
                retries +=1
                continue
            except json.JSONDecodeError:
                print(f"  错误: 解析查询结果JSON失败: {response.text}")
                time.sleep(5)
                retries +=1
                continue

            # print(f"  Polling result (attempt {retries+1}): {result_poll}") # Optional: for detailed poll debugging

            if 'content' in result_poll and 'orderInfo' in result_poll['content'] and 'status' in result_poll['content']['orderInfo']:
                status = result_poll['content']['orderInfo']['status']
            else:
                print("  查询结果格式不正确，无法获取订单状态。")
                status = -1 # Assume error status to break loop
            
            print(f"  当前状态: {status}")
            
            if status == 4: # Successfully processed
                final_result_json = result_poll
                break
            if status != 3: # Any other terminal state (e.g., -1 for error, 0, 1, 2 for other states)
                final_result_json = result_poll # Store the final (possibly error) state
                break 
            
            time.sleep(5) # Wait before next poll
            retries += 1
        
        if retries >= max_retries and status == 3:
            print("  错误: 查询转写结果超时。")
            return ""

        if not final_result_json:
            print("  错误: 未能获取最终转写结果。")
            return ""

        # print("  get_result 최종 resp:", final_result_json) # Optional: print final raw response

        full_text_segments = []
        if final_result_json and 'content' in final_result_json and 'orderResult' in final_result_json['content']:
            order_result_str = final_result_json['content']['orderResult']
            if order_result_str:
                try:
                    order_result_json = json.loads(order_result_str)
                    if 'lattice' in order_result_json:
                        for i, segment_data in enumerate(order_result_json['lattice']):
                            if 'json_1best' in segment_data:
                                json_1best_str = segment_data['json_1best']
                                try:
                                    json_1best_data = json.loads(json_1best_str)
                                    current_segment_text_parts = []
                                    if 'st' in json_1best_data and 'rt' in json_1best_data['st'] and isinstance(json_1best_data['st']['rt'], list):
                                        for part in json_1best_data['st']['rt']:
                                            if 'ws' in part and isinstance(part['ws'], list):
                                                for word_info in part['ws']:
                                                    if 'cw' in word_info and isinstance(word_info['cw'], list) and len(word_info['cw']) > 0:
                                                        if 'w' in word_info['cw'][0]:
                                                            current_segment_text_parts.append(word_info['cw'][0]['w'])
                                    segment_text = "".join(current_segment_text_parts)
                                    if segment_text:
                                        full_text_segments.append(segment_text)
                                except json.JSONDecodeError as e_inner:
                                    print(f"  错误: 解析分段 {i+1} 的 json_1best 失败: {e_inner}")
                                except TypeError as e_type:
                                    print(f"  错误: 解析分段 {i+1} 时出现类型错误: {e_type}")
                        
                        if full_text_segments:
                            concatenated_text = "".join(full_text_segments)
                            print(f"  识别到的文本: \"{concatenated_text[:100]}{'...' if len(concatenated_text) > 100 else ''}\"")
                            return concatenated_text
                        else:
                            print("  未提取到有效文本内容。")
                    else:
                        print("  未在 orderResult 中找到 'lattice' 字段。")
                except json.JSONDecodeError as e_outer:
                    print(f"  错误: 解析 orderResult 失败: {e_outer}")
            else:
                print("  orderResult 字段为空，无转写结果。")
        else:
            order_info = final_result_json.get('content', {}).get('orderInfo', {})
            final_status = order_info.get('status', '未知')
            fail_type = order_info.get('failType', '未知')
            desc_info = final_result_json.get('descInfo', '无描述')
            print(f"  未找到转写结果 (orderResult)。最终订单状态: {final_status}, 失败类型: {fail_type}, 描述: {desc_info}")
        
        return "" # Return empty string if no text extracted or error

def get_audio_duration(filepath):
    """
    Calculates the duration of a WAV audio file.
    Returns duration in seconds, or 0 if an error occurs.
    """
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except wave.Error as e:
        print(f"错误: 无法读取WAV文件 {filepath}: {e}")
        return 0
    except Exception as e:
        print(f"错误: 获取音频时长失败 {filepath}: {e}")
        return 0

def batch_transcribe_folder(appid, secret_key, audio_folder_path, output_text_file_path):
    """
    Transcribes all sentence_*.wav files in a folder and saves results.
    """
    print(f"开始批量转写任务...")
    print(f"音频文件夹: {audio_folder_path}")
    print(f"输出文件: {output_text_file_path}")

    # Find audio files matching the pattern sentence_*.wav and sort them
    search_pattern = os.path.join(audio_folder_path, "sentence_*.wav")
    audio_files = sorted(glob.glob(search_pattern))

    if not audio_files:
        print(f"在文件夹 {audio_folder_path} 中未找到匹配 'sentence_*.wav' 格式的音频文件。")
        return

    all_transcriptions = []
    total_processing_time_seconds = 0
    total_audio_duration_seconds = 0
    processed_files_count = 0
    
    overall_start_time = time.time()

    for i, audio_file_path in enumerate(audio_files):
        filename = os.path.basename(audio_file_path)
        print(f"\n--- 处理文件 {i+1}/{len(audio_files)}: {filename} ---")
        
        file_start_time = time.time()
        
        audio_duration = get_audio_duration(audio_file_path)
        if audio_duration == 0:
            print(f"  跳过文件 {filename} 因为无法获取其时长或时长为0。")
            all_transcriptions.append(f"Error: Could not process {filename} (duration error)\n")
            continue

        print(f"  音频时长: {audio_duration:.2f} 秒")
        total_audio_duration_seconds += audio_duration
        
        # Create a new API object for each file to get fresh ts and signa
        api = RequestApi(appid=appid,
                         secret_key=secret_key,
                         upload_file_path=audio_file_path,
                         audio_duration_seconds=audio_duration)
        
        transcribed_text = api.get_result() # This now returns the concatenated text or empty string
        
        if transcribed_text:
            all_transcriptions.append(transcribed_text + "\n") # Add newline after each file's transcription
            processed_files_count +=1
        else:
            all_transcriptions.append(f"Error: No transcription for {filename}\n")
            print(f"  文件 {filename} 未能成功转写或未返回文本。")

        file_end_time = time.time()
        file_processing_time = file_end_time - file_start_time
        total_processing_time_seconds += file_processing_time
        print(f"  文件 {filename} 处理耗时: {file_processing_time:.2f} 秒")

    overall_end_time = time.time()
    overall_total_time = overall_end_time - overall_start_time

    # Write results to output file
    try:
        with open(output_text_file_path, 'w', encoding='utf-8') as f:
            for line in all_transcriptions:
                f.write(line)
        print(f"\n--- 所有转写结果已写入: {output_text_file_path} ---")
    except IOError as e:
        print(f"错误: 无法写入输出文件 {output_text_file_path}: {e}")

    # --- Statistics ---
    print("\n--- 统计信息 ---")
    print(f"处理文件总数: {len(audio_files)}")
    print(f"成功转写文件数: {processed_files_count}")
    print(f"总音频时长: {total_audio_duration_seconds:.2f} 秒")
    print(f"总识别耗时: {total_processing_time_seconds:.2f} 秒")

    if processed_files_count > 0:
        avg_time_per_file = total_processing_time_seconds / processed_files_count
        print(f"平均每个成功转写文件的识别耗时: {avg_time_per_file:.2f} 秒")
    else:
        print("没有文件被成功转写，无法计算平均耗时。")

    if total_audio_duration_seconds > 0:
        rtf = total_processing_time_seconds / total_audio_duration_seconds
        print(f"整体实时率 (RTF): {rtf:.4f}")
    else:
        print("总音频时长为0，无法计算RTF。")
    
    print(f"整个批处理过程总耗时: {overall_total_time:.2f} 秒")


# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 请用户配置以下参数 ---
    YOUR_APPID = "YOUR_APPID"  # 替换为你的讯飞开放平台 APPID
    YOUR_SECRET_KEY = "YOUR_SECRET_KEY"  # 替换为你的讯飞开放平台 SECRET_KEY
    
    # !!重要!!: 请确保以下路径正确, 尤其是 APPID 和 SECRET_KEY
    # 如果 APPID 或 SECRET_KEY 不正确，API会返回认证失败的错误。
    
    # 示例路径 (请修改为你的实际路径)
    # Windows 示例: r"C:\Users\YourUser\Documents\AudioFiles"
    # Linux/macOS 示例: "/home/user/audio_files"
    INPUT_AUDIO_FOLDER = r"/root/audio-16k" # 替换为存放 sentence_*.wav 文件的文件夹路径
    OUTPUT_TEXT_FILE = r"all_transcriptions.txt" # 指定输出文本文件的路径和名称

    # --- 检查配置 ---
    if YOUR_APPID == "YOUR_ACTUAL_APPID" or YOUR_SECRET_KEY == "YOUR_ACTUAL_SECRETKEY":
        print("错误：请在脚本中设置您的讯飞 APPID 和 SECRET_KEY。")
        print("请修改 YOUR_APPID 和 YOUR_SECRET_KEY 变量的值。")
        exit()

    if not os.path.isdir(INPUT_AUDIO_FOLDER):
        print(f"错误：指定的音频文件夹不存在: {INPUT_AUDIO_FOLDER}")
        # 尝试创建示例文件和文件夹，如果需要测试脚本结构
        print("正在尝试创建示例音频文件夹和虚拟音频文件用于测试结构...")
        try:
            if not os.path.exists(INPUT_AUDIO_FOLDER):
                os.makedirs(INPUT_AUDIO_FOLDER)
                print(f"已创建示例文件夹: {INPUT_AUDIO_FOLDER}")
            
            # 创建一些虚拟的 .wav 文件 (这些不是真正的音频，仅用于测试文件查找和流程)
            for i in range(1, 4):
                dummy_file_path = os.path.join(INPUT_AUDIO_FOLDER, f"sentence_{i}.wav")
                if not os.path.exists(dummy_file_path):
                    # 创建一个最小的有效 WAV 文件头，然后加一点点数据
                    # 实际内容不重要，重要的是 get_audio_duration 能处理
                    try:
                        with wave.open(dummy_file_path, 'wb') as wf:
                            wf.setnchannels(1)      # Mono
                            wf.setsampwidth(2)      # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            wf.writeframes(b'\x00\x00' * 16000) # 1 second of silence
                        print(f"已创建虚拟音频文件: {dummy_file_path}")
                    except Exception as e_wave:
                        print(f"创建虚拟WAV文件失败 {dummy_file_path}: {e_wave}")
                        # If creating dummy files fails, just write a simple text file
                        with open(dummy_file_path, 'w') as f:
                             f.write("This is a dummy wav placeholder.")


            print("重要提示: 创建的虚拟文件不是有效的音频，仅用于演示脚本结构。")
            print("请确保 INPUT_AUDIO_FOLDER 中包含您要处理的真实 .wav 音频文件。")
        except Exception as e:
            print(f"创建示例文件夹或文件时出错: {e}")
            exit()
    
    # --- 运行批量转写 ---
    batch_transcribe_folder(appid=YOUR_APPID,
                            secret_key=YOUR_SECRET_KEY,
                            audio_folder_path=INPUT_AUDIO_FOLDER,
                            output_text_file_path=OUTPUT_TEXT_FILE)

    print("\n--- 批处理完成 ---")