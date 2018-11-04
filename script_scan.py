import os
import json


file_list = []


def scan_pics(path, tags, ignore_non_folder=False):
    items = os.listdir(path)
    for item in items:
        if item[0] == '.':
            continue
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            tags.append(item)
            scan_pics(new_path, tags)
            tags.pop()
        elif not ignore_non_folder:
            new_path = new_path.replace('\\', '/')
            print(new_path, tags)
            file_list.append(new_path)


if __name__ == '__main__':
    scan_pics('.', [], True)
    with open('list.json', 'w') as f:
        f.write(json.dumps(file_list))
