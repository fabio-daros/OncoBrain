import os
import json

# Parâmetros
json_input_path = './dataset/cric_detection/annotations/classifications.json'
output_json_path = './dataset/cric_detection/annotations/annotations_coco.json'
images_dir = './dataset/cric_detection/images'
box_size = 40  # Tamanho fixo do bounding box (ajuste conforme necessário)

# Mapeamento das classes para category_id COCO
CATEGORY_MAPPING = {
    "Negative for intraepithelial lesion": 1,
    "ASC-US": 2,
    "LSIL": 3,
    "ASC-H": 4,
    "HSIL": 5,
    "SCC": 6
}


def main():
    with open(json_input_path, 'r') as f:
        cric_data = json.load(f)

    images = []
    annotations = []
    categories = []
    ann_id = 1
    image_id_map = {}
    img_id = 1

    # Define categorias COCO
    for class_name, category_id in CATEGORY_MAPPING.items():
        categories.append({
            "id": category_id,
            "name": class_name,
            "supercategory": "cell"
        })

    for item in cric_data:
        image_filename = item['image_name']
        image_path = os.path.join(images_dir, image_filename)

        # Placeholder para tamanho da imagem (ajuste caso queira usar PIL para pegar real)
        width, height = 1024, 1024

        images.append({
            "id": img_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })
        image_id_map[image_filename] = img_id

        for cell in item['classifications']:
            x_center = cell['nucleus_x']
            y_center = cell['nucleus_y']
            x_min = max(x_center - box_size / 2, 0)
            y_min = max(y_center - box_size / 2, 0)
            width_box = box_size
            height_box = box_size

            class_label = cell['bethesda_system']
            category_id = CATEGORY_MAPPING.get(class_label, None)
            if category_id is None:
                print(f"Warning: Class '{class_label}' não encontrada no mapeamento. Pulando...")
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_json, f, indent=4)

    print(f"COCO JSON criado em: {output_json_path}")
    print(f"Total de imagens: {len(images)}")
    print(f"Total de anotações: {len(annotations)}")


if __name__ == "__main__":
    main()
