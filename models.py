import numpy as np
from scipy.optimize import linear_sum_assignment


class EasyModel:
    """
    Our first model for tracking smiles
    This model receives information about the bounding boxes at the frame
    and returns predicts labels for each of them
    At the first the model use the Hungarian algorithm with IOU values between trackings and detections.
    At the second -  if a detection don't match to any tracking,
    Euclidean distance is used.
    All not matching detections are assigned new labels.

    """

    def __init__(self,
                 iou_threshold: float = .15,
                 min_dist: int = 130) -> None:

        self.n_frame: int = 1
        self.last_bounding_boxes: dict = {}
        self.iou_threshold = iou_threshold
        self.label = 0
        self.min_dist = min_dist

    @staticmethod
    def calculate_iou(
            bound_box_1: list,
            bound_box_2: list) -> float:

        """
        Calculate the IOU metric between two bounding boxes.
        Coordinates are provided in the format (x,y,x,y).
        """

        b1_xmin = min(bound_box_1[0], bound_box_1[2])
        b1_xmax = max(bound_box_1[0], bound_box_1[2])
        b1_ymin = min(bound_box_1[1], bound_box_1[3])
        b1_ymax = max(bound_box_1[1], bound_box_1[3])

        b2_xmin = min(bound_box_2[0], bound_box_2[2])
        b2_xmax = max(bound_box_2[0], bound_box_2[2])
        b2_ymin = min(bound_box_2[1], bound_box_2[3])
        b2_ymax = max(bound_box_2[1], bound_box_2[3])

        s_b1 = (b1_xmax - b1_xmin) * (b1_ymax - b1_ymin)
        s_b2 = (b2_xmax - b2_xmin) * (b2_ymax - b2_ymin)

        # coordinates for intersection
        x_left = max(b1_xmin, b2_xmin)
        y_top = max(b1_ymin, b2_ymin)
        x_right = min(b1_xmax, b2_xmax)
        y_bottom = min(b1_ymax, b2_ymax)

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        union_area = s_b1 + s_b2 - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def create_iou_matrix(self, data: dict):
        """
        Creates the IOU matrix for the intersections between the bounding boxes
        from the previous frame and the ones found on the current frame.
        """
        # drop empty elements from data
        data = [el for el in data if len(el['bounding_box']) > 0]
        len_new = len(data)
        len_pred = len(self.last_bounding_boxes)

        # Created dict with key for tracking
        keys_trackers = {}
        for index, key in enumerate(self.last_bounding_boxes):
            keys_trackers[index] = key

        #
        keys_detection = {}
        for index, value in enumerate(data):
            keys_detection[index] = value

        iou_matrix = np.zeros((len_pred, len_new))

        for i, key in enumerate(self.last_bounding_boxes):
            for j, elements in enumerate(data):
                iou_value = self.calculate_iou(self.last_bounding_boxes[key], elements['bounding_box'])
                if iou_value > self.iou_threshold:
                    iou_matrix[i, j] = iou_value

        return iou_matrix, keys_detection, keys_trackers

    @staticmethod
    def euclid_distance(b1: list,
                        b2: list) -> float:
        """
        Calculates the Euclidean distance between two points.
        b1, b2 - A list in the format [x_min, y_min, x_max, y_max].
        """
        x1, y1 = (b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2
        x2, y2 = (b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2

        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return dist

    def predict(self, metadata: list) -> None:
        """
        Predicts the labels of tracked objects.
        On the first frame, it initializes the labels randomly.
        After used the Hungarian algorithm and Euclidean distance for tracking.
        """

        # If first time:
        # Create a dict that assigns labels to all find bounding boxes.
        # If bounding boxes is None, skip it.
        if self.n_frame == 1:
            for i in range(len(metadata["data"])):
                if len(metadata["data"][i]['bounding_box']) != 0:
                    self.last_bounding_boxes[self.label] = metadata["data"][i]['bounding_box']
                    metadata["data"][i]['track_id'] = self.label
                    self.label += 1

            self.n_frame += 1

            return metadata

        # Next time:
        # We have the coordinates of all detected objects before.
        # Create a matrix for the Hungarian algorithm with IOU values.
        data = metadata["data"]
        iou_matrix, keys_detection, keys_trackers = self.create_iou_matrix(data)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind):
            # If this value is greater than 0, this is a new bounding box for that label.
            if iou_matrix[i, j] > 0:
                matches.append((i, j))

        # The matches variable contains pairs
        # A pair like (0,3) means that the new bounding box for keys_last_bb[0] is keys_new_bb[3].
        # If a label is missing in matches[0], it means that we couldn't find a new bounding box for that label,
        # so we keep using the old one.
        # If a label is missing in matches[1], it means that this is a bounding box for a new object,
        # so we add it to our list.

        # update tracking labels with new bb
        upd_labels = []
        for i, j in matches:
            # Update the bounding boxes based on the old labels.
            self.last_bounding_boxes[keys_trackers[i]] = keys_detection[j]['bounding_box']
            upd_labels.append(keys_trackers[i])
            # Update the label for the new detection that we consider to correspond to it.
            for v in range(len(metadata["data"])):
                if metadata["data"][v]['bounding_box'] == keys_detection[j]['bounding_box']:
                    metadata["data"][v]['track_id'] = keys_trackers[i]

        # Let's calculate the distance between  detections and tracking labels that weren't use.
        # If min distance is less than self.min_dist, assign this detection to the old label.

        for key in self.last_bounding_boxes:
            min_dist = 10 ** 8
            if key in upd_labels:
                continue
            for ind in range(len(metadata["data"])):
                if metadata["data"][ind]['track_id'] is None and len(metadata["data"][ind]['bounding_box']):
                    dist = self.euclid_distance(self.last_bounding_boxes[key], metadata["data"][ind]['bounding_box'])
                    if dist < min_dist:
                        min_dist = dist
                        n_el = ind
            if min_dist < self.min_dist:
                self.last_bounding_boxes[key] = metadata["data"][n_el]['bounding_box']
                metadata["data"][n_el]['track_id'] = key

        # All undefined detections - will use as new labels.
        # We record them in our dictionary.

        for ind in range(len(metadata["data"])):
            if metadata["data"][ind]['track_id'] is None and len(metadata["data"][ind]['bounding_box']):
                metadata["data"][ind]['track_id'] = self.label
                self.last_bounding_boxes[self.label] = metadata["data"][ind]["bounding_box"]
                self.label += 1
        return metadata

