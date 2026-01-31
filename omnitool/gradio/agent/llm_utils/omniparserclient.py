import requests
import base64
from pathlib import Path
from tools.screen_capture import get_screenshot
from agent.llm_utils.utils import encode_image

OUTPUT_DIR = "./tmp/outputs"

class OmniParserClient:
    def __init__(self, 
                 url: str) -> None:
        self.url = url

    def __call__(self,):
        screenshot, screenshot_path = get_screenshot()
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        print(f"[DEBUG] Sending request to: {self.url}")
        print(f"[DEBUG] Image base64 length: {len(image_base64)}")

        try:
            response = requests.post(self.url, json={"base64_image": image_base64}, timeout=120)
            print(f"[DEBUG] Response status code: {response.status_code}")
            print(f"[DEBUG] Response headers: {response.headers}")

            if response.status_code != 200:
                print(f"[ERROR] OmniParser server returned error {response.status_code}")
                print(f"[ERROR] Response text: {response.text}")
                raise Exception(f"OmniParser server error {response.status_code}: {response.text}")

            response_json = response.json()
            print('omniparser latency:', response_json['latency'])
        except requests.exceptions.JSONDecodeError as e:
            print(f"[ERROR] Failed to decode JSON response")
            print(f"[ERROR] Response text: {response.text}")
            raise Exception(f"Invalid JSON response from OmniParser server: {response.text}") from e
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request to OmniParser server failed: {str(e)}")
            raise Exception(f"Failed to connect to OmniParser server at {self.url}") from e

        som_image_data = base64.b64decode(response_json['som_image_base64'])
        screenshot_path_uuid = Path(screenshot_path).stem.replace("screenshot_", "")
        som_screenshot_path = f"{OUTPUT_DIR}/screenshot_som_{screenshot_path_uuid}.png"
        with open(som_screenshot_path, "wb") as f:
            f.write(som_image_data)
        
        response_json['width'] = screenshot.size[0]
        response_json['height'] = screenshot.size[1]
        response_json['original_screenshot_base64'] = image_base64
        response_json['screenshot_uuid'] = screenshot_path_uuid
        response_json = self.reformat_messages(response_json)
        return response_json
    
    def reformat_messages(self, response_json: dict):
        screen_info = ""
        for idx, element in enumerate(response_json["parsed_content_list"]):
            element['idx'] = idx
            if element['type'] == 'text':
                screen_info += f'ID: {idx}, Text: {element["content"]}\n'
            elif element['type'] == 'icon':
                screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
        response_json['screen_info'] = screen_info
        return response_json