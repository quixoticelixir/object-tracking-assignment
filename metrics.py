import pandas as pd


class CustomTrackingMetric:
    """
    We have real_label and our customs label on each detection
    We are creating a dataset with predicted custom labels for each frame.
    Then we calculate the ratio of the number of matches for the first (correct) custom label
    to the total number of label names for this detection.
    Then we take the average of all probabilities.
    """

    def __init__(self):
        self.data = []

    def add_row(self, predict_for_frame: dict) -> None:
        """
        Add row in our self.data
        param predict_for_frame: dict with true label and custom label (cb_id, track_id)
        """

        row = {'frame_id': predict_for_frame['frame_id']}
        for el in predict_for_frame['data']:
            row[f"cb_id_{el['cb_id']}"] = el['track_id']

        self.data.append(row)

    def calculate_metric(self) -> float:
        """
        Method which calculate our custom metric
        """
        data = pd.DataFrame(self.data)
        data = data.drop(["frame_id"], axis=1)
        macthing_labels = data.apply(lambda x: x[x.notnull()].iloc[0])

        detect_percent_for_label = []
        for col in data.columns:
            count_all_detected = data[col].count()
            count_true_detected = sum(data[col] == macthing_labels[col])
            percent_for_label = count_true_detected / count_all_detected
            detect_percent_for_label.append(percent_for_label)

        return round(sum(detect_percent_for_label) / len(detect_percent_for_label), 3)


# Quick Check
if __name__ == '__main__':
    from models import EasyModel

    easy_model = EasyModel()
    metric_model = CustomTrackingMetric()
    from track_26_04_18_10 import track_data

    for el in track_data:
        predict = easy_model.predict(el)

        metric_model.add_row(predict)

    metrics = metric_model.calculate_metric()
    print(metrics)