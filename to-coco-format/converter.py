#!/usr/bin/python3

import pandas as pd
from sys import argv
from glob import glob
from json import dumps

if len(argv) < 2:
    print("USAGE: ./converter.py DATASET_PATH")
    exit(1)

def category_conversion(input_cat: int) -> int:
    """
    Our categories:
    none, paragraph, title, code, ul, ol
    """
    if input_cat == 5:
        return 6
    return int(input_cat)

out = {
    "annotations": [],
    "images": [],
    "categories": [
        {"supercategory": "","id": 1, "name": "text"},
        {"supercategory": "", "id": 2, "name": "title"},
        {"supercategory": "", "id": 3, "name": "code"},
        {"supercategory": "", "id": 4, "name": "ul"},
        {"supercategory": "", "id": 5, "name": "figure"},
        {"supercategory": "", "id": 6, "name": "ol"},
    ]
}
annotation_id = 0
for i, f in enumerate(glob(argv[1] + '/*.csv')):
    df = pd.read_csv(f, dtype=int)
    # add the image to the image list
    image_id = i # alternativa: int.from_bytes(f.encode())
    
    out["images"].append(
        {
            "width": 1700,
            "height": 2200,
            "id": image_id,
            "file_name": f.rstrip('.csv').replace("-bb", "").replace(".png", ".jpg")
        }
    )
    for _, el in df.iterrows():
        out["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "area": (el['End X']-el['Start X']) * (el['End Y']-el['Start Y']),
                "category_id": category_conversion(el["Type"]),
                "iscrowd": 0,
                "bbox": [ # x,y,width,height
                    el['Start X'],
                    el['Start Y'],
                    el['End X']-el['Start X'],
                    el['End Y']-el['Start Y'],
                ],
                "segmentation": [ # polygon format: list of x,y coordinates of vertices
                    [
                        el['Start X'],
                        el['Start Y'],
                        el['End X'],
                        el['Start Y'],
                        el['End X'],
                        el['End Y'],
                        el['Start X'],
                        el['End Y'],
                    ]
                ]
            }
        )
        annotation_id+=1

print(dumps(out, default=int))
