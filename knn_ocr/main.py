import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread


data = Path("C:/users/User/Workspace/Education/4_sem/AI-CV/HW/knn_ocr/task")
train_path = data / "train"


def extractor(image):
	if image.ndim == 2: # если картинка уже серая
		gray = image.astype("u1")
	else:
		gray = np.mean(image, 2).astype("u1")

	binary = gray > 0 # объект не черный пиксель
	lb = label(binary)
	props = regionprops(lb)

	if len(props) == 0: # если нету объектов, то пустой вектор признаков
		return np.zeros(5, dtype="f4")

	prop = max(props, key=lambda p: p.area)
	area = max(float(prop.area), 1.0)
	convex_area = max(float(prop.area_convex), 1.0)

	features = [
		float(prop.eccentricity), # вытянутость фигуры
		float(prop.perimeter) / area,
		float(prop.extent), # насколько фигура заполняет свой прямоугольник. Sфигуры / Sbbox
		float(prop.solidity), # насколько фигура цельная. отношение площади объекта к площади его выпуклой оболочки
		convex_area / area, # насколько фигура не выпуклая
	]

	return np.array(features, dtype="f4")


def make_train(path):
	train = []
	responses = []
	classes = []
	ncls = 0

	for cls in sorted(path.glob("*")):
		if not cls.is_dir():
			continue
		ncls += 1
		classes.append(cls.name[-1])
		for p in sorted(cls.glob("*.png")):
			train.append(extractor(imread(p)))
			responses.append(ncls)

	train = np.array(train, dtype="f4").reshape(-1, 5)
	responses = np.array(responses, dtype="f4").reshape(-1, 1)
	return train, responses, classes


def predict_(image, knn, k=5):
	feat = extractor(image).reshape(1, -1)
	ret, results, neighbours, dist = knn.findNearest(feat, k)
	return ret, results, neighbours, dist


def get_test_images(path):
	return sorted(path.glob("*.png"))


def split_to_components(image):
	if image.ndim == 2:
		gray = image.astype("u1")
	else:
		gray = np.mean(image, 2).astype("u1")

	binary = gray > 0 # объект не черный пиксель
	lb = label(binary)
	props = regionprops(lb)
	if len(props) == 0:
		return []

	heights = [p.bbox[2] - p.bbox[0] for p in props]
	median_h = float(np.median(heights)) if len(heights) > 0 else 0.0 # медианная высота фигуры
	min_h = max(3, int(median_h * 0.35)) # минимальная допустимая высота компоненты
	min_area = max(10, int(median_h * 0.25)) # минимальная допустимая площадь компоненты

	components = []
	for prop in props:
		h = prop.bbox[2] - prop.bbox[0]
		if prop.area < min_area: # пропуск мелкого мума по площади
			continue
		if h < min_h: # пропуск слишком низких компонент
			continue
		row0, col0, row1, col1 = prop.bbox # координаты bbox
		crop = binary[row0:row1, col0:col1].astype("u1") # вырезание bbox из бинарного изображения
		components.append({
            "bbox": prop.bbox, # расположение фигуры на исходном изображении
            "image": crop, # вырезанная фигура
            "area": prop.area # площадь вырезанной фигуры
        })

	components = sorted(components, key=lambda x: x["bbox"][1]) # сортировка по координате x(col0) для правильного порядка букв
	return components


def merge_pair_components(comp_a, comp_b):
	row0a, col0a, row1a, col1a = comp_a["bbox"] # bbox первой компоненты
	row0b, col0b, row1b, col1b = comp_b["bbox"] # bbox второй компоненты

	row0 = min(row0a, row0b) # общий верх
	col0 = min(col0a, col0b) # общий левый край
	row1 = max(row1a, row1b) # общий низ
	col1 = max(col1a, col1b) # общий правый край

	space = np.zeros((row1 - row0, col1 - col0), dtype="u1") # область под объединенный символ
	space[
        row0a - row0: row1a - row0,
        col0a - col0: col1a - col0
    ] = np.maximum(space[
            row0a - row0: row1a - row0,
            col0a - col0: col1a - col0
        ], comp_a["image"], # перенос первой компоненты на область
    )
	space[
        row0b - row0: row1b - row0,
        col0b - col0: col1b - col0
    ] = np.maximum(space[
            row0b - row0: row1b - row0,
            col0b - col0: col1b - col0
        ], comp_b["image"], # перенос второй компоненты на область
	)

	return {
		"bbox": (row0, col0, row1, col1), # bbox объединенной компоненты
		"image": space, # бинарное изображение объединенного символа
		"area": int(space.sum()), # площадь объединенной компоненты
	}


def merge_two_part_components(components):
	if len(components) < 2:
		return components

	heights = [c["bbox"][2] - c["bbox"][0] for c in components]
	median_h = float(np.median(heights)) if len(heights) > 0 else 1.0 # типичная высота компоненты

	merged = [] # итоговый список после возможной склейки
	i = 0
	while i < len(components):
		if i == len(components) - 1: # последняя компонента без пары
			merged.append(components[i])
			break

		curr = components[i]
		next_comp = components[i + 1]

		row0, col0, row1, col1 = curr["bbox"]
		next_row0, next_col0, next_row1, next_col1 = next_comp["bbox"]
		w1, w2 = col1 - col0, next_col1 - next_col0

		x_overlap = min(col1, next_col1) - max(col0, next_col0) # пересечение по оси X
		overlap_ok = x_overlap > 0 and x_overlap >= 0.55 * min(w1, w2) # достаточное горизонтальное пересечение
		close_x = abs((col0 + col1) / 2.0 - (next_col0 + next_col1) / 2.0) <= max(w1, w2) * 0.3 # центры близко по X

		v_gap_1 = next_row0 - row1 # next ниже curr
		v_gap_2 = row0 - next_row1 # curr ниже next
		vertical_ok = \
            (0 <= v_gap_1 <= median_h * 0.45) or \
            (0 <= v_gap_2 <= median_h * 0.45) # компоненты близки по вертикали

		if overlap_ok and close_x and vertical_ok:
			merged.append(merge_pair_components(curr, next_comp)) # склейка двухэлементного символа
			i += 2 # пропуск соединенной пары
			continue

		merged.append(curr)
		i += 1

	return merged


def recognize_components_text(components, knn, classes, k=3):
	text = "" # строка результат по компонентам слева направо
	if len(components) == 0:
		return text

	widths = [c["bbox"][3] - c["bbox"][1] for c in components] # ширина каждой компоненты (символа)
	median_w = float(np.median(widths)) if len(widths) > 0 else 0.0 # медианная ширина символа
	space_gap = max(2, int(median_w * 0.55)) # порог вставки пробела между символами

	for curr_idx, comp in enumerate(components):
		ret, _, _, _ = predict_(comp["image"], knn, k) # предсказание класса для одной компоненты
		idx = int(ret) - 1 # переход от класса 1..N к индексу 0..N-1
		if 0 <= idx < len(classes):
			text += classes[idx]
		else:
			text += "?" # защита от некорректного индекса

		if curr_idx < len(components) - 1:
			next_comp = components[curr_idx + 1]
			curr_right = comp["bbox"][3]
			next_left = next_comp["bbox"][1]
			gap = next_left - curr_right
			if gap > space_gap:
				text += " " # добавление пробела при большом разрыве
	return text


if __name__ == "__main__":
	train, responses, classes = make_train(train_path)

	knn = cv2.ml.KNearest.create()
	knn.train(train, cv2.ml.ROW_SAMPLE, responses)

	print(f"{classes=}")
	for test_image in get_test_images(data):
		image = imread(test_image)
		components = split_to_components(image)
		components = merge_two_part_components(components)
		text = recognize_components_text(components, knn, classes, 3)
		print(f"{test_image.stem}: {text}")
