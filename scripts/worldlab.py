import requests
import json
import glob
import argparse
import os

MARBLE_API_KEY = os.getenv("MARBLE_API_KEY")

def process_worldlab(input_type = "video", input_path = "/home/haotian/polaris/sofa.mp4", prompt = "a sofa", display_name = "my sofa"):

    headers = {
            "WLT-Api-Key": MARBLE_API_KEY,
            "Content-Type": "application/json"
        }
    
    generate_url = "https://api.worldlabs.ai/marble/v1/worlds:generate"
    payload = {}
    
    if input_type == "video":
        media_url = "https://api.worldlabs.ai/marble/v1/media-assets:prepare_upload"

        payload = {
            "file_name": input_path,
            "kind": input_type,
            "extension": "mp4"
        }
        
        response = requests.post(media_url, json=payload, headers=headers)
        print(response.text)

        data = json.loads(response.text)

        upload_info = data["upload_info"]
        signed_upload_url = upload_info["upload_url"]
        headers = upload_info["required_headers"]
        media_asset_id = data["media_asset"]["media_asset_id"]

        with open(input_path, 'rb') as f:
            video_data = f.read()

        requests.put(
            signed_upload_url,
            headers=headers,
            data=video_data
        )

        payload = {
            "display_name": display_name,
            "world_prompt": {
                "type": input_type,
                "video_prompt": {
                    "source": "media_asset",
                    "media_asset_id": media_asset_id
                },
                "text_prompt": prompt
            }
        }
        headers = {
            "WLT-Api-Key": "K3XkQSyIjHB7Nc0pCkoNc9lrO2niCzB5",
            "Content-Type": "application/json"
        }
        response = requests.post(generate_url, json=payload, headers=headers)
        print(response.text)

        operation_id = json.loads(response.text)["operation_id"]
        url = f"https://api.worldlabs.ai/marble/v1/worlds/{operation_id}"

        headers = {"WLT-Api-Key": "K3XkQSyIjHB7Nc0pCkoNc9lrO2niCzB5",}

        response = requests.get(url, headers=headers)

        print(response.text)


    elif input_type == "multi-image":

        def upload_image(file_path, file_name):
            # Prepare upload
            prepare_response = requests.post(
                'https://api.worldlabs.ai/marble/v1/media-assets:prepare_upload',
                headers={
                    'WLT-Api-Key': 'YOUR_API_KEY',
                    'Content-Type': 'application/json'
                },
                json={
                    'file_name': file_name,
                    'kind': 'image',
                    'extension': 'jpg'
                }
            )

            data = prepare_response.json()
            media_asset = data['media_asset']
            upload_info = data['upload_info']

            # Upload file
            with open(file_path, 'rb') as f:
                requests.put(
                    upload_info['upload_url'],
                    headers=upload_info['required_headers'],
                    data=f.read()
                )

            return media_asset['id']

        image_paths = sorted(glob.glob(f"{input_path}/*.jpg"))
        media_asset_idx = []

        for i in range(len(image_paths), 10):
            import os
            file_name = os.path.basename(image_paths[i])
            media_asset_idx.append(upload_image(image_paths[i], file_name))
        
        multi_image_prompt = []
        for id in media_asset_idx:

            multi_image_prompt.append({
                "azimuth": 0,
                "content": {
                    "source": "media_asset",
                    "media_asset_id": id
                }
            })

        payload = {
            "display_name": display_name,
            "world_prompt": {
                "type": input_type,
                "multi_image_prompt": multi_image_prompt,
                "text_prompt": prompt
            }
        }

    elif input_type == "image":

        payload = {
            "file_name": input_path,
            "kind": "image",
            "extension": "jpg"
        }

        response = requests.post(url, json=payload, headers=headers)
        print(response.text)

        data = json.loads(response.text)

        upload_info = data["upload_info"]
        signed_upload_url = upload_info["upload_url"]
        headers = upload_info["required_headers"]
        media_asset_id = data["media_asset"]["media_asset_id"]

        with open(input_path, 'rb') as f:
            image_data = f.read()

        requests.put(
            signed_upload_url = upload_info["upload_url"],
            headers=upload_info['required_headers'],
            data=image_data
        )

        payload = {
            "display_name": display_name,
            "world_prompt": {
                "type": "image",
                "image_prompt": {
                    "source": "media_asset",
                    "media_asset_id": media_asset_id
                },
                "text_prompt": prompt
            }
        }
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run world lab.")
    
    parser.add_argument(
        "--input_type",
        help="video, multi-image, or image",
        type=str,
        default="video",
    )
    parser.add_argument("--input_path", default="/home/haotian/polaris/sofa_v2.mp4", help="Path to input files.")
    parser.add_argument("--prompt", default="a sofa", help="Text prompt to guide the world generation.")
    parser.add_argument("--display_name", default="my sofa", help="Display name for the generated world.")
    args = parser.parse_args()
    process_worldlab(args.input_type, args.input_path, args.prompt, args.display_name)