import os, json, shutil

"""
This skript filters the COCO images according to our use case.
Only single person images are kept.
"""

coco_annotation_json_path = "/Users/josephinevandelden/Downloads/" \
                            "annotations/person_keypoints_val2017.json"
image_source_dir_path = "/Users/josephinevandelden/Downloads/val2017"
dst_coco_annotation_json_path = "/testing/annotation.json"
dst_image_dir_path = "/testing/images_1person"

with open(coco_annotation_json_path, 'r') as f:
    annotation_json_info = json.loads(f.read())

print(annotation_json_info.keys())
print(annotation_json_info['categories'])
print(annotation_json_info['info'])

image_infos = annotation_json_info['images']
annotation_infos = annotation_json_info['annotations']

print()
print("=" * 80)
print(annotation_infos[0])
# dict_keys: 'segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints',
# 'image_id', 'bbox', 'category_id', 'id'
print(annotation_infos[0].keys())

annotation_infos_by_image_id = {}
for annotation_info in annotation_infos:
    image_id = annotation_info['image_id']
    if image_id in annotation_infos_by_image_id:
        annotation_infos_by_image_id[image_id].append(annotation_info)
    else:
        annotation_infos_by_image_id[image_id] = [annotation_info]

image_ids = list(annotation_infos_by_image_id.keys())
maximum_anntated_num = max(list(
    map(lambda image_id: len(annotation_infos_by_image_id[image_id]),
        image_ids)))
minimum_anntated_num = min(list(
    map(lambda image_id: len(annotation_infos_by_image_id[image_id]),
        image_ids)))

print("max:", maximum_anntated_num, "min:", minimum_anntated_num)

print()

pnum_and_count = list(map(lambda num: (num, len(list(
    filter(lambda image_id: len(annotation_infos_by_image_id[image_id]) == num,
           image_ids)))), range(minimum_anntated_num,
                                maximum_anntated_num + 1)))
for person_num, image_num in pnum_and_count:
    print("", person_num, "->", image_num)

print("=" * 80)

image_id_to_image_info = {}
for image_info in image_infos:
    image_id_to_image_info[image_info['id']] = image_info

print("=" * 80)

single_person_image_ids = list(
    filter(
        lambda image_id: len(
            annotation_infos_by_image_id[image_id]) == 1, image_ids))
print(len(single_person_image_ids))

print()

filtered_json_annotation_info = {}
filtered_json_annotation_info['categories'] = annotation_json_info['categories']
# image_infos
filtered_image_infos = list(map(
    lambda image_id: image_id_to_image_info[image_id], single_person_image_ids))
filtered_json_annotation_info['images'] = filtered_image_infos
print(len(filtered_image_infos))
# annotation_infos
filterted_annotation_infos = list(map(
    lambda image_id: annotation_infos_by_image_id[image_id][0],
    single_person_image_ids))
filtered_json_annotation_info['annotations'] = filterted_annotation_infos
print(len(filterted_annotation_infos))

print()
print("images num of new:", len(filtered_json_annotation_info['images']))
print("annots num of new:", len(filtered_json_annotation_info['annotations']))

for image_info in filtered_json_annotation_info['images']:
    if not os.path.exists(os.path.join(
            image_source_dir_path, image_info['file_name'])):
        print(f"ERR: no image file in "
              f"{os.path.join(image_source_dir_path, image_info['file_name'])}")
        exit(0)
print("============ NO error for file existing check ============")
print()

# write annotation.json
print("=" * 80)
print("=" * 80)
print(f"WRITE START AT {dst_coco_annotation_json_path}")
with open(dst_coco_annotation_json_path, 'w') as fp:
    json.dump(filtered_json_annotation_info, fp)
print(f"WRITE END AT {dst_coco_annotation_json_path}")
print("=" * 80)
print("=" * 80)

print()

# copy image files
echo_num = 100
pass_num = 0
copy_num = 0
total_num = len(filtered_json_annotation_info['images'])
print(f"START COPYING {total_num} FILES")
for idx, image_info in enumerate(filtered_json_annotation_info['images']):
    src_image_path = os.path.join(image_source_dir_path,
                                  image_info['file_name'])
    dst_image_path = os.path.join(dst_image_dir_path, image_info['file_name'])
    if not os.path.exists(dst_image_path):
        shutil.copyfile(src_image_path, dst_image_path)
        copy_num += 1
    else:
        pass_num += 1

    if (idx + 1) % echo_num == 0:
        print(f"  >> {idx + 1} / {total_num}, copy:{copy_num}, pass:{pass_num}")
print(f"END COPYING {total_num} FILES, copy:{copy_num}, pass:{pass_num}")
print("=" * 80)
print("=" * 80)