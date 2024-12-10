import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import locality_aware_nms as nms_locality
import traceback
import time


class TextDetector:
    def __init__(self, saved_model_path, max_side_len=512, resize_factor=2):
        self.RESIZE_FACTOR = resize_factor
        self.max_side_len = max_side_len
        self.model = self._load_model(saved_model_path)

    def _load_model(self, saved_model_path):
        custom_objects = {"RESIZE_FACTOR": self.RESIZE_FACTOR, }
        try:
            model = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects)
            print("Successfully loaded text detection model")
            return model
        except Exception as e:
            traceback.print_exc()
            return None

    def detect_text(self, img):
        if self.model is None:
            print("Model not loaded!")
            return None

        start_time = time.time()
        img_resized, (ratio_h, ratio_w) = self._resize_image(img)

        img_resized = (img_resized / 127.5) - 1
        timer = {'net': 0, 'restore': 0, 'nms': 0}

        start = time.time()
        score_map, geo_map = self.model.predict(img_resized[np.newaxis, :, :, :])
        timer['net'] = time.time() - start

        boxes, timer = self._detect(
            score_map=score_map,
            geo_map=geo_map,
            timer=timer,
            score_map_thresh=0.8,
            box_thresh=0.1,
            nms_thres=0.2
        )

        print('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        data = self._process_boxes(boxes, img)
        df = pd.DataFrame(data)
        if not df.empty:
            df = self._sort_bounding_boxes(df)
        return df

    def _resize_image(self, img):
        h, w, _ = img.shape
        resize_h, resize_w = (self.max_side_len, self.max_side_len)
        img_resized = cv2.resize(img, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img_resized, (ratio_h, ratio_w)

    def _detect(self, score_map, geo_map, timer, score_map_thresh, box_thresh, nms_thres):
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, :]

        xy_text = np.argwhere(score_map > score_map_thresh)
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        start = time.time()
        text_box_restored = self._restore_rectangle_rbox(
            xy_text[:, ::-1] * 4,
            geo_map[xy_text[:, 0], xy_text[:, 1], :]
        )
        print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        timer['restore'] = time.time() - start

        start = time.time()
        boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        timer['nms'] = time.time() - start

        if boxes.shape[0] == 0:
            return None, timer

        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]
        return boxes, timer

    def _restore_rectangle_rbox(self, origin, geometry):
        d = geometry[:, :4]
        angle = geometry[:, 4]
        # for angle > 0
        origin_0 = origin[angle >= 0]
        d_0 = d[angle >= 0]
        angle_0 = angle[angle >= 0]
        if origin_0.shape[0] > 0:
            p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                        d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                        d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                        np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                        d_0[:, 3], -d_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                    new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))
        # for angle < 0
        origin_1 = origin[angle < 0]
        d_1 = d[angle < 0]
        angle_1 = angle[angle < 0]
        if origin_1.shape[0] > 0:
            p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                        np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                        np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                        -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                        -d_1[:, 1], -d_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                    new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))
        return np.concatenate([new_p_0, new_p_1])

    def _process_boxes(self, boxes, img):
        data = []
        if boxes is not None:
            for box in boxes:
                box = self._sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                points = box.astype(np.int32).reshape((-1, 1, 2))
                x, y, w, h = cv2.boundingRect(points)
                # cv2.rectangle(img[:, :, ::-1], (x, y), (x + w, y + h), (255, 0, 0), 2)

                data.append({"x": x, "y": y, "w": w, "h": h})
        return data

    def _sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def _sort_bounding_boxes(self, df, row_threshold=10):
        # Sort by y to group rows
        df = df.sort_values(by='y', ascending=True).reset_index(drop=True)
        
        # Assign row groups based on y-coordinate
        row_groups = [0]  # First row starts as 0
        current_row = 0
        previous_y = df.loc[0, 'y']

        for i in range(1, len(df)):
            if abs(df.loc[i, 'y'] - previous_y) > row_threshold:
                current_row += 1
            row_groups.append(current_row)
            previous_y = df.loc[i, 'y']
        
        df['row_group'] = row_groups

        # Sort within each row group by x-coordinate
        df = df.sort_values(by=['row_group', 'x']).reset_index(drop=True)

        # Drop the row_group column
        df = df.drop(columns=['row_group'])
        return df
