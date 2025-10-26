import json
from collections import Counter

#19 - people 25 - tables
# Путь к файлу COCO JSON
json_file = "/home/timofey/Documents/projects/table_assist/data/new_roboflow_datapack/Restaurant_Tables.v1i.coco/valid/_annotations.coco.json"

# Загружаем JSON
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Общая информация
images = data.get('images', [])
annotations = data.get('annotations', [])
categories = data.get('categories', [])

print(f"Всего изображений: {len(images)}")
print(f"Всего аннотаций (объектов): {len(annotations)}")
print(f"Всего классов: {len(categories)}")

# Список названий классов
category_id_to_name = {cat['id']: cat['name'] for cat in categories}
print("Список классов:")
for cat_id, cat_name in category_id_to_name.items():
    print(f"{cat_id}: {cat_name}")

# Сколько объектов каждого класса
category_counts = Counter()
for ann in annotations:
    category_counts[ann['category_id']] += 1

print("\nКоличество объектов по классам:")
for cat_id, count in category_counts.items():
    print(f"{category_id_to_name[cat_id]}: {count}")

# Пример: среднее количество объектов на изображение
avg_objects_per_image = len(annotations) / len(images) if images else 0
print(f"\nСреднее количество объектов на изображение: {avg_objects_per_image:.2f}")
