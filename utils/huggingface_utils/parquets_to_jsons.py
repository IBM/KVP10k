def parquets_to_jsons(parquets):
    id_fields = ['image_url', 'page_number', 'hash_name']
    annotations_fields = ['cuboids', 'landmarks', 'lines', 'polygons', 'rectangles']

    assert len(parquets) % 5 == 0, "Data problem, abort."
    n_jsons = len(parquets) // 5
    jsons = []
    for i in (range(n_jsons)):
        anchor_json = parquets[5 * i]
        current_json = {}
        current_json["annotations"] = {annotations_fields[0]: anchor_json["annotations"]}
        for field in id_fields:
            current_json[field] = anchor_json[field]
        for j in range(1, 5):
            anchor_json = parquets[5 * i + j]
            for field in id_fields:
                assert anchor_json[field] == current_json[field], "Data problem, abort."
            current_json["annotations"][annotations_fields[j]] = anchor_json["annotations"]
        jsons.append(current_json)

    return jsons