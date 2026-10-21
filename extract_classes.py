import json
import os

def extract_classes_to_txt(json_filepath, txt_filepath):
    if not os.path.exists(json_filepath):
        print(f"[错误] 找不到 JSON 文件: {json_filepath}")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    class_names = []

    if isinstance(data, list):
        class_names = data
        
    elif isinstance(data, dict):
        if 'categories' in data and isinstance(data['categories'], list):
            if len(data['categories']) > 0 and isinstance(data['categories'][0], dict):
                class_names = [cat['name'] for cat in data['categories'] if 'name' in cat]
            else:
                class_names = data['categories']
                
        elif all(isinstance(v, str) for v in data.values()):
            class_names = list(data.values())
            
        else:
            print("[错误] 字典结构复杂，无法自动确定类别名称的键名！")
            return
    else:
        print("[错误] 无法识别的 JSON 基础结构（不是列表也不是字典）！")
        return

    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{str(name).strip()}\n")

    print(f"🎉 成功提取了 {len(class_names)} 个类别名称！")
    print(f"📁 已保存至: {os.path.abspath(txt_filepath)}")

if __name__ == '__main__':
    input_json = '/data/ranwanwu/Dataset/OVSIS/Vaihingen.json'
    output_txt = '/data/ranwanwu/Dataset/OVSIS/Vaihingen_classnames.txt'
    
    extract_classes_to_txt(input_json, output_txt)