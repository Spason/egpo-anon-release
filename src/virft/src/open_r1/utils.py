import base64
import os
import tempfile
import traceback
import urllib.request

import requests
from PIL import Image


def url_to_base64(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Request failed, status code: {response.status_code}")
            return None
    except Exception:
        print(f"Request failed, image_url: {image_url}")
        return None
    image_data = response.content
    base64_str = base64.b64encode(image_data).decode("utf-8")
    return base64_str


def extract_single_video_frames_to_url(video_file, frame_num, output_dir):
    """
    Extract frames from a video file or URL, save as image files, and return file path list.

    Args:
        video_file (str): Video file path or URL
        frame_num (int): Number of frames to extract
        output_dir (str): Output directory path

    Returns:
        list: List of image file paths
    """
    from moviepy.editor import VideoFileClip

    temp_file_path = None
    is_url = video_file.startswith('http://') or video_file.startswith('https://')
    frame_paths = []

    try:
        if is_url:
            print(f"Downloading remote video: {video_file}")
            temp_file_path = tempfile.mktemp(suffix='.mp4')
            urllib.request.urlretrieve(video_file, temp_file_path)
            video_file = temp_file_path

        if not os.path.exists(video_file):
            print(f"Video file does not exist: {video_file}")
            return frame_paths

        if os.path.getsize(video_file) == 0:
            print(f"File is empty or corrupted: {video_file}")
            return frame_paths

        os.makedirs(output_dir, exist_ok=True)

        with VideoFileClip(video_file) as clip:
            duration = clip.duration
            frame_times = [duration * i / frame_num for i in range(frame_num)]

            for i, t in enumerate(frame_times):
                filename = os.path.basename(video_file).split(".")[0] + f"_{i+1:04d}.jpg"
                save_path = os.path.join(output_dir, filename)

                if os.path.exists(save_path):
                    frame_paths.append(save_path)
                    continue

                frame_array = clip.get_frame(t)
                img = Image.fromarray(frame_array)
                img.save(save_path, "JPEG", quality=95)
                frame_paths.append(save_path)

        return frame_paths

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        traceback.print_exc()
        return []
    finally:
        if is_url and temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def url_array2base64_array(img_url):
    image_base64 = []
    for img_url_tmp in img_url:
        b64 = url_to_base64(img_url_tmp)
        if b64:
            image_base64.append("data:image/jpeg;base64," + b64)
    return image_base64
